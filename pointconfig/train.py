import heapq
import numpy as np
import matplotlib.pyplot as plt

from pointconfig.model import model_info, train_model
from pointconfig.plot import (
    plot_beginning,
    plot_middle_raw,
    plot_middle_normalized,
    plot_end,
)
from pointconfig.make_subset import generate_subsets, get_highest_subsets
from pointconfig.expand_subset import expand_subsets
from pointconfig.lightweight_score import (
    BATCH_SIZE,
    score_thresholds,
    PRIME,
    TOTAL_DIRECTIONS,
)


def training(loops=5000, top_examples=100, plot=True, checkpoint=True):
    if plot:
        fig, ax_top, ax_bottom = plot_beginning()

    loop_nums = []
    loop_nums_with_multiplicity = []
    all_best_scores = []
    best_scores_list = []
    mean_scores_list = []
    median_scores_list = []
    normalized_scores_list = []

    thresholds = score_thresholds(PRIME)
    max_threshold = max(thresholds)
    normalization_factor = PRIME * TOTAL_DIRECTIONS

    best_examples_seen = []

    # Assign fixed colors to each threshold
    threshold_labels = list(thresholds.items())
    fixed_colors = dict(
        zip(thresholds.keys(), plt.get_cmap("tab10").colors[: len(thresholds)])
    )

    model, loss_function, optimizer = model_info()
    for loop_num in range(loops):
        model.eval()
        all_subsets, scores = generate_subsets(model, BATCH_SIZE)
        best_subsets, best_scores = get_highest_subsets(
            all_subsets, scores, 90
        )

        for score, subset in zip(best_scores, best_subsets):
            beginning = len(best_examples_seen) < top_examples
            good_score = score > best_examples_seen[0][0]
            if beginning:
                heapq.heappush(best_examples_seen, (score, subset))
            elif good_score:
                heapq.heappushpop(best_examples_seen, (score, subset))

        training_set = expand_subsets(best_subsets)
        loss = train_model(
            training_set, model, loss_function, optimizer, shuffle=True
        )

        print(f"Loop {loop_num+1}, Loss: {loss}")
        print(f"Best scores mean: {best_scores.mean()}")

        best_score = best_scores.max().item()
        mean_score = best_scores.mean().item()
        best_scores_list.append(best_score)
        mean_scores_list.append(mean_score)
        median_scores_list.append(np.median(best_scores))
        normalized_score = (best_score - max_threshold) / normalization_factor
        normalized_scores_list.append(normalized_score)

        scores_this_loop = best_scores.tolist()
        loop_nums.append(loop_num)
        loop_nums_with_multiplicity.extend([loop_num] * len(scores_this_loop))
        all_best_scores.extend(scores_this_loop)

        if plot:
            plot_middle_raw(
                ax_top,
                loop_nums,
                loop_nums_with_multiplicity,
                all_best_scores,
                best_scores_list,
                mean_scores_list,
                median_scores_list,
            )
            plot_middle_normalized(
                ax_bottom,
                loop_nums,
                normalized_scores_list,
                max_threshold,
                normalization_factor,
                threshold_labels,
                fixed_colors,
            )

    if plot:
        plot_end()
