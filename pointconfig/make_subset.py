import torch
import numpy as np
import time

import pointconfig.lightweight_score as lws
import pointconfig.model as pcm

INPUT_LENGTH = 2 * lws.WORD_LENGTH


def generate_subset(model):
    all_stages = np.zeros((lws.WORD_LENGTH, INPUT_LENGTH + 1), dtype=np.uint8)
    for stage in range(lws.WORD_LENGTH):
        input_sequence = np.zeros(INPUT_LENGTH, dtype=np.uint8)
        for point_index in range(stage):
            input_sequence[point_index] = all_stages[stage - 1, point_index]
        input_sequence[stage + lws.WORD_LENGTH] = 1
        all_stages[stage, stage + lws.WORD_LENGTH] = 1

        prob = pcm.model_forward(input_sequence, model)
        include_point = np.uint8(np.random.rand() < prob)
        all_stages[stage, INPUT_LENGTH] = include_point
        for next_stage in range(stage + 1, lws.WORD_LENGTH):
            all_stages[next_stage, stage] = include_point

        subset = np.zeros(lws.WORD_LENGTH, dtype=np.uint8)
        for index in range(lws.WORD_LENGTH - 1):
            subset[index] = all_stages[-1, index]
        subset[-1] = all_stages[-1][-1]
        score = lws.score_word(subset)
    return all_stages, score


def generate_subsets(model, num_subsets):
    all_subsets = torch.zeros((num_subsets, INPUT_LENGTH), dtype=torch.float32)

    for stage in range(lws.WORD_LENGTH):
        #print(f"working on stage {stage + 1} out of {lws.WORD_LENGTH}")
        #t0 = time.time()
        all_subsets[:, stage + lws.WORD_LENGTH] = 1.0
        #t1 = time.time()
        probs = pcm.model_forward(all_subsets, model)
        #t2 = time.time()
        all_subsets[:, stage] = (torch.rand(num_subsets) < probs).to(
            torch.float32
        )
        #t3 = time.time()
        all_subsets[:, stage + lws.WORD_LENGTH] = 0.0
        #t4 = time.time()
        #print(
        #    f"Stage {stage:4d} | forward: {t1-t0:.4f}s | sample: {t2-t1:.4f}s | write: {t3-t2:.4f}s | fill future: {t4-t3:.4f}s"
        #)

    scores = lws.score_words(all_subsets.numpy().astype(np.uint8))

    return all_subsets, scores


def get_highest_subsets(subsets, scores, percentile):
    assert len(scores) == lws.BATCH_SIZE
    k_highest = int(lws.BATCH_SIZE * (1 - (percentile / 100)))
    indices = np.argpartition(scores, -k_highest)[-k_highest:]
    return subsets[torch.tensor(indices), :], scores[indices]
