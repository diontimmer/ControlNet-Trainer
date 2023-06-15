from share import create_controlnet_model
from cldm.model import create_model
import torch
import os
import argparse


def reattach_base(input_model, sd_version="2.1"):
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.

    base_model_path, base_model_config = create_controlnet_model(sd_version=sd_version)

    # load initial checkpoint

    print("Loading base model.")

    ini_model = torch.load(base_model_path, map_location=torch.device("cpu"))

    # load input model

    print("Loading input model.")

    input_model_torch = torch.load(input_model, map_location=torch.device("cpu"))

    # replace keys inside initial checkpoint with keys from input model

    print("Replacing keys in base model with keys from input model.")

    for k, v in input_model_torch.items():
        ini_model[k] = v

    model = create_model(base_model_config).cpu()
    model.load_state_dict(ini_model)

    # save model

    input_filename = os.path.basename(input_model)

    out_filename = f"base_{input_filename}"

    # add pytorch-lightning_version 0.0.0 to state_dict

    final_mod = {}

    final_mod["state_dict"] = model.state_dict()
    final_mod["pytorch-lightning_version"] = "0.0.0"
    final_mod["epoch"] = 0
    final_mod["global_step"] = 0
    final_mod["callbacks"] = {}
    final_mod["optimizer_states"] = {}
    final_mod["lr_schedulers"] = {}

    torch.save(final_mod, out_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model",
        type=str,
        required=True,
        help="Path to the model checkpoint to reattach the base.",
    )
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        help="The version of the base model to use.",
    )
    args = parser.parse_args()

    reattach_base(args.input_model, sd_version=args.sd_version)
