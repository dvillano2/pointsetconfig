import heapq
import numpy as np


class TrainingTracker:
    def __init__(self, num_top_examples):
        self.tracking_lists = self.__class__.make_tracking_lists()
        self.num_top_examples = num_top_examples
        self.top_examples = []

    def update_lists(self, best_scores, threshold_data, loop_num):
        update_info = self.__class__.make_update_info(
            best_scores, threshold_data
        )
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
            str_subset = "".join([str(inout) for inout in subset.tolist()])
            if len(self.top_examples) < self.num_top_examples:
                heapq.heappush(
                    self.top_examples,
                    (score, str_subset),
                )
            elif score > self.top_examples[0][0]:
                heapq.heappushpop(
                    self.top_examples,
                    (score, str_subset),
                )

    @staticmethod
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

    @staticmethod
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
