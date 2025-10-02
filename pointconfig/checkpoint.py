from pathlib import Path
from datetime import datetime
import os
import json
import torch
from torch import nn
from pointconfig.model import (
    model_info,
    FIRST_LAYER,
    SECOND_LAYER,
    THIRD_LAYER,
    LEARNING_RATE,
    INPUT_LENGTH,
)
from pointconfig.trainingtracker import TrainingTracker
from torch.serialization import add_safe_globals


def checkpoint(
    loop_num,
    complete_model_info,
    training_tracker,
    first_save,
    save_path=None,
    plot=None,
):
    if not first_save and save_path is None:
        raise ValueError("If its not the first save, provide save_path")

    if not save_path:
        training_path = Path("./training_runs")
        day_time = datetime.now().strftime("%b%d%Y_%H_%M_%S")
        save_path = training_path / day_time
        save_path.mkdir(exist_ok=True)
    else:
        save_path = Path(save_path)

    # file name info
    str_loop_num = str(loop_num)
    digits = len(str_loop_num)
    file_name_end = "_at_" + ("0" * (7 - digits)) + str_loop_num + "_loops"

    # save the figures
    if plot:
        figures_path = save_path / "figures"
        figures_path.mkdir(exist_ok=True)
        plot_name = "plot" + file_name_end + ".png"
        plot.savefig(figures_path / plot_name)

    # save the model and auxiliary info
    save_dict = {
        "loop_num": loop_num,
        "model": complete_model_info["model"].state_dict(),
        "optimizer": complete_model_info["optimizer"].state_dict(),
        "training_tracker": training_tracker,
    }
    old_model_name = ""
    for filename in os.listdir(save_path):
        if filename.startswith("model"):
            old_model_name = filename
    model_name = "model" + file_name_end + ".pt"
    torch.save(save_dict, save_path / model_name)
    if old_model_name:
        os.remove(save_path / old_model_name)

    # save the best examples as a json
    top_examples_save = {}
    for i, pair in enumerate(training_tracker.top_examples):
        score, subset = pair
        top_examples_save[i] = {"score": score, "subset": subset}
    best_examples_path = save_path / "top_examples.json"
    with open(best_examples_path, "w", encoding="utf8") as f:
        json.dump(top_examples_save, f, indent=4)

    return save_path


def load_model(path):
    model = nn.Sequential(
        nn.Linear(INPUT_LENGTH, FIRST_LAYER),
        nn.ReLU(),
        nn.Linear(FIRST_LAYER, SECOND_LAYER),
        nn.ReLU(),
        nn.Linear(SECOND_LAYER, THIRD_LAYER),
        nn.ReLU(),
        nn.Linear(THIRD_LAYER, 1),
        nn.Sigmoid(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    add_safe_globals([TrainingTracker])
    checkpoint_info = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint_info["model"])
    optimizer.load_state_dict(checkpoint_info["optimizer"])
    return (
        model,
        optimizer,
        checkpoint_info["loop_num"],
        checkpoint_info["training_tracker"],
    )

def load_checkpoint(save_path, top_examples):
    if save_path is not None:
        save_path = Path(save_path)
        for filename in os.listdir(save_path):
            if filename.startswith("model"):
                model_file = filename
        model_path = save_path / model_file
        model, optimizer, base_loop_num, training_tracker = load_model(
            model_path
        )
        loss_function = nn.BCELoss()
        complete_model_info = {
            "model": model,
            "loss_function": loss_function,
            "optimizer": optimizer,
        }
    else:
        complete_model_info = model_info()
        training_tracker = TrainingTracker(num_top_examples=top_examples)
        base_loop_num = 0

    return complete_model_info, training_tracker, base_loop_num
