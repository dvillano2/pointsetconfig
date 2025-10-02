import json
from argparse import ArgumentParser
from pointconfig.word_to_point import word_to_point, check_equidistribution
from pointconfig.lightweight_score import PRIME, DIMENSION


def check_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        top_examples = json.load(f)
    word_list = [pair["subset"] for pair in top_examples.values()]
    pointset_list = [
        word_to_point(word, PRIME, DIMENSION) for word in word_list
    ]
    return [
        len(check_equidistribution(pointset, PRIME, DIMENSION))
        for pointset in pointset_list
    ]


def main():
    parser = ArgumentParser()
    parser.add_argument("--examples_path")
    args = parser.parse_args()
    if not args.examples_path:
        raise ValueError("need to provide json path after --examples_path")
    print(sorted(check_from_json(args.examples_path)))


if __name__ == "__main__":
    main()
