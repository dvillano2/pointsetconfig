from pointconfig.model import model_info, train_model
from pointconfig.make_subset import (
    generate_subset,
    generate_subsets,
    get_highest_subsets,
)
from pointconfig.lightweight_score import BATCH_SIZE
from pointconfig.expand_subset import expand_subsets
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    loop_nums = []
    loop_nums_with_multiplicity = []
    all_best_scores = []
    best_scores_list = []
    mean_scores_list = []
    median_scores_list = []

    model, loss_function, optimizer = model_info()
    for loop_num in range(10000):
        model.eval()
        all_subsets, scores = generate_subsets(model, BATCH_SIZE)
        best_subsets, best_scores = get_highest_subsets(
            all_subsets, scores, 90
        )
        training_set = expand_subsets(best_subsets)
        loss = train_model(
            training_set, model, loss_function, optimizer, shuffle=True
        )

        print(f"Loop {loop_num+1}, Loss: {loss}")
        print(f"Best scores mean: {best_scores.mean()}")

        # Store stats
        all_best_scores_list = best_scores.tolist()
        loop_nums.append(loop_num)
        loop_nums_with_multiplicity.extend(
            [loop_num] * len(all_best_scores_list)
        )
        all_best_scores.extend(all_best_scores_list)
        best_scores_list.append(best_scores.max().item())
        mean_scores_list.append(best_scores.mean().item())
        median_scores_list.append(np.median(best_scores))

        # Clear and redraw plot
        ax.clear()
        ax.scatter(
            loop_nums_with_multiplicity,
            all_best_scores,
            alpha=0.1,
            color="gray",
            s=10,
            label="All Scores",
        )
        ax.plot(loop_nums, best_scores_list, label="Best Score")
        ax.plot(loop_nums, mean_scores_list, label="Mean Score")
        ax.plot(loop_nums, median_scores_list, label="Median Score")
        ax.set_xlabel("Loop Number")
        ax.set_ylabel("Score")
        ax.set_title("Training Progress")
        ax.legend()
        plt.pause(0.01)

    plt.ioff()  # Turn off interactive mode when done
    plt.show()
