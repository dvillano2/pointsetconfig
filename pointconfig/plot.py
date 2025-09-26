import matplotlib.pyplot as plt


def plot_beginning():
    """Brief set up of the plot"""
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
    """Plots the raw score on the top"""
    ax_top.clear()

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
    """Plots the normalized scores on the bottom"""
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
    """Finishes off the plotting"""
    plt.ioff()
    plt.tight_layout()
    plt.show()
