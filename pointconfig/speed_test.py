import time
import numpy as np
import pointconfig.lightweight_score as lws


def profile_score_word():
    max_score = -(lws.PRIME**5)
    length = lws.WORD_LENGTH
    runs = 10000000
    times = np.empty(runs)
    for run in range(runs):
        word = np.zeros(length)
        num_non_trivial_points = (5 * 11) - 4
        point_indices = np.random.choice(
            length, num_non_trivial_points, replace=False
        )
        for index in point_indices:
            word[index] = 1

        start = time.perf_counter()
        score = lws.score_word(word)
        end = time.perf_counter()
        max_score = max(score, max_score)
        times[run] = end - start
        if (run + 1) % 100000 == 0:
            print(score)
            percentage_done = 100 * ((run + 1) / runs)
            print(f"Current Score: {score}")
            print(f"Percentage done: {percentage_done:.2f} %")
            print(f"Max Score: {max_score}")
            mean_time_so_far = np.mean(times[: run + 1])
            print(f"Mean time to score so far is {mean_time_so_far:.8f}\n\n")
            print("\n\n\n\n")


if __name__ == "__main__":
    profile_score_word()
