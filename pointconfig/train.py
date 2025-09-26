import heapq
import numpy as np
import matplotlib.pyplot as plt

from pointconfig.model import model_info, train_model
from pointconfig.plot import (
    plot_beginning,
    plot_middle,
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


def make_tracking_lists():
    tracking_lists = {
        "loop_nums": [],
        "loop_nums_with_multiplicity": [],
        "all_best_scores": [],
        "best_scores_list": [],
        "mean_scores_list": [],
        "median_scores_list": [],
        "normalized_scores_list": [],
    }
    return tracking_lists


def make_thresholds_and_data(prime, total_directions):
    thresholds = score_thresholds(prime)
    threshold_data = {
        "max_threshold": max(thresholds),
        "normalization_factor": prime * total_directions,
        "threshold_labels": list(thresholds.items()),
        # Assign fixed colors to each threshold
        "fixed_colors": dict(
            zip(
                thresholds.keys(),
                plt.get_cmap("tab10").colors[: len(thresholds)],
            )
        ),
    }
    return threshold_data


def make_update_info(best_scores, threshold_data):
    update_info = {
        "best_score": best_scores.max().item(),
        "mean_score": best_scores.mean().item(),
        "scores_this_loop": best_scores.tolist(),
    }
    update_info["normalized_score"] = (
        update_info["best_score"] - threshold_data["max_threshold"]
    ) / threshold_data["normalization_factor"]
    return update_info


def training(loops=5000, top_examples=100, plot=True, checkpoint=True):
    """Trains, plots, and checkpoints"""
    if plot:
        fig, ax_top, ax_bottom = plot_beginning()

    tracking_lists = make_tracking_lists()
    threshold_data = make_thresholds_and_data(PRIME, TOTAL_DIRECTIONS)
    best_examples_seen = []

    model, loss_function, optimizer = model_info()
    for loop_num in range(loops):
        model.eval()
        all_subsets, scores = generate_subsets(model, BATCH_SIZE)
        best_subsets, best_scores = get_highest_subsets(
            all_subsets, scores, 90
        )

        for score, subset in zip(best_scores, best_subsets):
            if len(best_examples_seen) < top_examples:
                heapq.heappush(best_examples_seen, (score, subset))
            elif score > best_examples_seen[0][0]:
                heapq.heappushpop(best_examples_seen, (score, subset))

        training_set = expand_subsets(best_subsets)
        loss = train_model(
            training_set, model, loss_function, optimizer, shuffle=True
        )

        print(f"Loop {loop_num+1}, Loss: {loss}")
        print(f"Best scores mean: {best_scores.mean()}")

        update_info = make_update_info(best_scores, threshold_data)

        tracking_lists["best_scores_list"].append(update_info["best_score"])
        tracking_lists["mean_scores_list"].append(update_info["mean_score"])
        tracking_lists["median_scores_list"].append(np.median(best_scores))
        tracking_lists["normalized_scores_list"].append(
            update_info["normalized_score"]
        )
        tracking_lists["loop_nums"].append(loop_num)
        tracking_lists["loop_nums_with_multiplicity"].extend(
            [loop_num] * len(update_info["scores_this_loop"])
        )
        tracking_lists["all_best_scores"].extend(
            update_info["scores_this_loop"]
        )

        if plot:
            plot_middle(ax_top, ax_bottom, tracking_lists, threshold_data)

    if plot:
        plot_end()
