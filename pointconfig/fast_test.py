import time
import random
from pointconfig.subset import Subset  # replace with your actual import


def profile_add_points(prime: int, dimension: int, num_points: int) -> None:
    # Generate unique points first
    points = set()
    while len(points) < num_points:
        pt = tuple(random.randint(0, prime - 1) for _ in range(dimension))
        points.add(pt)

    subset = Subset(prime, dimension)

    start = time.perf_counter()
    for pt in points:
        subset.add_point(pt)
    end = time.perf_counter()

    print(f"Added {num_points} distinct points in {end - start:.4f} seconds")


def main():
    prime = 41
    dimension = 3
    num_points = prime**2  # adjust as needed
    profile_add_points(prime, dimension, num_points)


if __name__ == "__main__":
    main()
