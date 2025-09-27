import heapq
import numpy as np
import matplotlib.pyplot as plt

from pointconfig.trainingtracker import TrainingTracker
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


def best_from_model(model, batch_size, percentile=90):
    """returns a tuple of best subset and best scores"""
    all_subsets, scores = generate_subsets(model, batch_size)
    return get_highest_subsets(all_subsets, scores, percentile)


def train(loops=5000, top_examples=100, plot=True):
    """Trains, plots, and checkpoints"""
    if plot:
        fig, ax_top, ax_bottom = plot_beginning()

    training_tracker = TrainingTracker(num_top_examples=top_examples)
    threshold_data = make_thresholds_and_data(PRIME, TOTAL_DIRECTIONS)
    complete_model_info = model_info()

    for loop_num in range(loops):
        complete_model_info["model"].eval()
        best_subsets, best_scores = best_from_model(
            complete_model_info["model"], BATCH_SIZE
        )
        training_tracker.update_best_examples(best_subsets, best_scores)

        # training_tracker.update_best_examples(best_subsets, best_scores)
        training_set = expand_subsets(best_subsets)
        training_tracker.tracking_lists["loss"].append(
            train_model(
                training_set,
                complete_model_info["model"],
                complete_model_info["loss_function"],
                complete_model_info["optimizer"],
                shuffle=True,
            )
        )

        print(
            f"At {loop_num+1}, Loss: "
            f"{training_tracker.tracking_lists['loss'][-1]}"
        )
        print(f"Best scores mean: {best_scores.mean()}")

        training_tracker.update_lists(best_scores, threshold_data, loop_num)

        if plot:
            plot_middle(
                ax_top,
                ax_bottom,
                training_tracker.tracking_lists,
                threshold_data,
            )

    if plot:
        plot_end()


if __name__ == "__main__":
    train()
