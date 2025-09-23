import torch

import pointconfig.lightweight_score as lws

INPUT_LENGTH = 2 * lws.WORD_LENGTH


def expand_subset(subset: torch.Tensor):
    repeats = torch.unsqueeze(subset, 1).expand(-1, lws.WORD_LENGTH)
    intermediate = torch.tril(repeats, diagonal=-1)
    stage_tracker = torch.eye(lws.WORD_LENGTH)
    return torch.cat([intermediate, stage_tracker], dim=1)


def expand_subsets(subsets: torch.Tensor):
    assert subsets.shape[1] == INPUT_LENGTH
    subsets = subsets[:, : lws.WORD_LENGTH]
    good_batch_size = subsets.shape[0]
    stage_tracker = torch.eye(lws.WORD_LENGTH).expand(good_batch_size, -1, -1)
    stages = subsets.unsqueeze(1).expand(-1, lws.WORD_LENGTH, -1)
    tri_stages = stages.tril(diagonal=-1)
    truth = stages.diagonal(dim1=1, dim2=2).unsqueeze(2)
    return torch.cat([tri_stages, stage_tracker, truth], dim=2).reshape(
        good_batch_size * lws.WORD_LENGTH, INPUT_LENGTH + 1
    )
