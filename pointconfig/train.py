import argparse
from torch import nn
from pointconfig.checkpoint import checkpoint, load_checkpoint
from pointconfig.model import train_model
from pointconfig.plot import (
    plot_beginning,
    plot_middle,
    plot_end,
    make_thresholds_and_data,
)
from pointconfig.make_subset import generate_subsets, get_highest_subsets
from pointconfig.expand_subset import expand_subsets
from pointconfig.lightweight_score import (
    BATCH_SIZE,
    PRIME,
    TOTAL_DIRECTIONS,
)


def best_from_model(model, batch_size, percentile=90):
    """returns a tuple of best subset and best scores"""
    all_subsets, scores = generate_subsets(model, batch_size)
    return get_highest_subsets(all_subsets, scores, percentile)


def train(
    loops=5000,
    top_examples=100,
    plot=True,
    save_checkpoint=True,
    save_path=None,
):
    """Trains, plots, and checkpoints"""
    if plot:
        fig, ax_top, ax_bottom = plot_beginning()
    else:
        fig = None

    threshold_data = make_thresholds_and_data(PRIME, TOTAL_DIRECTIONS)
    complete_model_info, training_tracker, base_loop_num = load_checkpoint(
        save_path, top_examples
    )
    first_save = bool(save_path is None)
    for loop_num in range(loops):
        complete_model_info["model"].eval()
        best_subsets, best_scores = best_from_model(
            complete_model_info["model"], BATCH_SIZE
        )
        training_tracker.update_best_examples(best_subsets, best_scores)

        training_set = expand_subsets(best_subsets)
        loss = train_model(
            training_set,
            complete_model_info["model"],
            complete_model_info["loss_function"],
            complete_model_info["optimizer"],
            shuffle=True,
        )
        training_tracker.tracking_lists["loss"].append(loss)
        print(
            f"At {base_loop_num + loop_num + 1}, Loss: "
            f"{training_tracker.tracking_lists['loss'][-1]}\n"
            f"Best scores mean: {best_scores.mean()}"
        )

        training_tracker.update_lists(
            best_scores, threshold_data, base_loop_num + loop_num
        )

        if plot:
            plot_middle(
                ax_top,
                ax_bottom,
                training_tracker.tracking_lists,
                threshold_data,
            )

        if save_checkpoint and loop_num % 50 == 49:
            save_path = checkpoint(
                base_loop_num + loop_num + 1,
                complete_model_info,
                training_tracker,
                first_save,
                save_path,
                fig,
            )
            first_save = False

    if plot:
        plot_end(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    args = parser.parse_args()
    if args.model_path:
        train(plot=False, save_path=args.model_path)
    else:
        train(plot=False)


if __name__ == "__main__":
    main()
