from pathlib import Path
from datetime import datetime
import json
import torch


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
        day_time = datetime.now().strftime("%d%b%Y_%H_%M_%S")
        save_path = training_path / day_time
        save_path.mkdir(exist_ok=True)

    # file name info
    str_loop_num = str(loop_num)
    digits = len(str_loop_num)
    file_name_end = "_at_" + ("0" * (5 - digits)) + str_loop_num + "_loops"

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
    model_name = "model" + file_name_end + ".pt"
    torch.save(save_dict, save_path / model_name)

    # save the best examples as a json
    top_examples_save = {}
    for i, pair in enumerate(training_tracker.top_examples):
        score, subset = pair
        top_examples_save[i] = {"score": score, "subset": subset}
    best_examples_path = save_path / "top_exmples.json"
    with open(best_examples_path, "w", encoding="utf8") as f:
        json.dump(top_examples_save, f)

    return save_path
