import heapq
import numpy as np
import matplotlib.pyplot as plt

from pointconfig.model import model_info, train_model
from pointconfig.make_subset import generate_subsets, get_highest_subsets
from pointconfig.expand_subset import expand_subsets
from pointconfig.lightweight_score import (
    BATCH_SIZE,
    score_thresholds,
    PRIME,
    TOTAL_DIRECTIONS,
)


def plot_beginning():
    plt.ion()
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True
    )
    return fig, ax_top, ax_bottom


def plot_middle_raw(
    ax_top,
    loop_nums,
    loop_nums_with_multiplicity,
    all_best_scores,
    best_scores_list,
    mean_scores_list,
    median_scores_list,
):
    # --- Plotting ---

    ax_top.clear()

    # Top plot: raw scores
    ax_top.scatter(
        loop_nums_with_multiplicity,
        all_best_scores,
        alpha=0.1,
        color="gray",
        s=10,
        label="All Scores",
    )
    ax_top.plot(loop_nums, best_scores_list, label="Best Score")
    ax_top.plot(loop_nums, mean_scores_list, label="Mean Score")
    ax_top.plot(loop_nums, median_scores_list, label="Median Score")

    ax_top.set_ylabel("Raw Score")
    ax_top.set_title("Training Score Progress")
    ax_top.legend(loc="upper left", fontsize=8)


def plot_middle_normalized(
    ax_bottom,
    loop_nums,
    normalized_scores_list,
    max_threshold,
    normalization_factor,
    threshold_labels,
    fixed_colors,
):
    ax_bottom.clear()

    # Bottom plot: normalized scores
    ax_bottom.plot(
        loop_nums,
        normalized_scores_list,
        label="Normalized Score",
        color="purple",
    )

    # Plot normalized threshold lines
    threshold_lines = []
    threshold_line_labels = []
    for threshold_val, label in threshold_labels:
        normalized_val = (threshold_val - max_threshold) / normalization_factor
        color = fixed_colors[threshold_val]
        line = ax_bottom.axhline(
            normalized_val,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        threshold_lines.append(line)
        threshold_line_labels.append(label)

    # Integer level lines (light gray)
    max_norm = int(max(normalized_scores_list + [0])) + 1
    for level in range(max_norm):
        ax_bottom.axhline(level, color="gray", linestyle="--", alpha=0.2)
        ax_bottom.text(
            loop_nums[-1] if loop_nums else 0,
            level,
            f"{level}",
            fontsize=8,
            alpha=0.5,
            verticalalignment="bottom",
            horizontalalignment="right",
        )

    ax_bottom.set_xlabel("Loop Number")
    ax_bottom.set_ylabel("Normalized Score")
    ax_bottom.set_title(
        "Normalized Score (Positive integers count equidistribution)"
    )
    ax_bottom.legend(
        threshold_lines + [ax_bottom.lines[0]],
        threshold_line_labels + ["Normalized Score"],
        fontsize=8,
        loc="lower right",
    )

    plt.pause(0.01)


def plot_end():
    plt.ioff()
    plt.tight_layout()
    plt.show()


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
            top_n = bool(
                best_examples_seen and score > best_examples_seen[0][0]
            )
            if beginning or top_n:
                heapq.heappush((score, subset))

        training_set = expand_subsets(best_subsets)
        loss = train_model(
            training_set, model, loss_function, optimizer, shuffle=True
        )

        print(f"Loop {loop_num+1}, Loss: {loss}")
        print(f"Best scores mean: {best_scores.mean()}")

        best_score = best_scores.max().item()
        best_scores_list.append(best_score)
        mean_scores_list.append(best_scores.mean().item())
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
