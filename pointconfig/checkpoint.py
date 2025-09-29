from pathlib import Path
import torch
from datetime import datetime
import json


def checkpoint(
    loop_num,
    complete_model_info,
    training_tracker,
    best_examples,
    first_save,
    save_path=None,
    plot=None,
):
    if not first_save and save_path is None:
        raise ValueError("If its not the first save, provide save_path")

    if not save_path:
        training_path = Path("./training_runs")
        day_time = datetime.now().strftime("%d%b%Y_%H_%M_%S")
        save_path = training_path / day_time
        save_path.mkdir(exists_ok=True)

    # save the figures
    if plot:
        figures_path = save_path / "figures"
        figures_path.mkdir(exists_ok=True)
        plot.savefig(figures_path)

    # save the model and auxiliary info
    model_path = save_path / "model"
    model_path.mkdir(exists_ok=True)
    save_dict = {
        "loop_num": loop_num,
        "model": complete_model_info["model"].state_dict(),
        "optimizer": complete_model_info["optimizer"].state_dict(),
        "training_tracker": training_tracker,
    }
    torch.save(save_dict)

    # save the best examples as a json
    best_examples_save = {}
    for i, pair in enumerate(best_examples):
        score, subset = pair
        best_examples_save[i] = {"score": score, "subset": subset}
    best_examples_path = save_path / "best_exmples.json"
    json.dump(best_examples_save, best_examples_path)

    return save_path
