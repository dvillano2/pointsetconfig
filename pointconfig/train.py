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


def make_tracking_lists():
    tracking_lists = {
        "loop_nums": [],
        "loop_nums_with_multiplicity": [],
        "all_best_scores": [],
        "best_scores_list": [],
        "mean_scores_list": [],
        "median_scores_list": [],
        "normalized_scores_list": [],
        "loss": [],
    }
    return tracking_lists


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


class TrainingTracker:
    def __init__(self, num_top_examples):
        self.tracking_lists = make_tracking_lists()
        self.best_examples_seen = []
        self.num_top_examples = num_top_examples
        self.top_examples = []

    def update_lists(self, best_scores, threshold_data, loop_num):
        update_info = make_update_info(best_scores, threshold_data)
        self.tracking_lists["best_scores_list"].append(
            update_info["best_score"]
        )
        self.tracking_lists["mean_scores_list"].append(
            update_info["mean_score"]
        )
        self.tracking_lists["median_scores_list"].append(
            np.median(best_scores)
        )
        self.tracking_lists["normalized_scores_list"].append(
            update_info["normalized_score"]
        )
        self.tracking_lists["loop_nums"].append(loop_num)
        self.tracking_lists["loop_nums_with_multiplicity"].extend(
            [loop_num] * len(update_info["scores_this_loop"])
        )
        self.tracking_lists["all_best_scores"].extend(
            update_info["scores_this_loop"]
        )

    def update_best_examples(self, best_subsets, best_scores):
        for score, subset in zip(best_scores, best_subsets):
            if len(self.top_examples) < self.num_top_examples:
                heapq.heappush(
                    self.top_examples,
                    (score, subset),
                )
            elif score > self.top_examples[0][0]:
                heapq.heappushpop(
                    self.top_examples,
                    (score, subset),
                )


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
