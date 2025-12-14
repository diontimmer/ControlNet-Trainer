import config
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from safetensors.torch import save_file
from types import SimpleNamespace
import torch
import sys
import json
import os
import urllib.request
from tqdm import tqdm


# config is the defaults. read the first sys argument to read the config json and update the dict
def make_config():
    if len(sys.argv) > 1 and os.path.splitext(sys.argv[1])[1] == ".json":
        print("Loading config from json:", sys.argv[1])
        json_path = sys.argv[1]
        # load json and cast to python dict with python types
        with open(json_path, "rt", encoding="utf-8") as f:
            new_config = json.load(f)
        # update config with the new dict
        for k, v in new_config.items():
            config.config[k] = v

        config.config = SimpleNamespace(**config.config)


make_config()
config = config.config


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        state_dict = torch.load(filepath, map_location="cpu")
        try:
            state_dict = state_dict["state_dict"]["state_dict"]
        except:
            try:
                state_dict = state_dict["state_dict"]
            except:
                pass

        if any([k.startswith("control_model.") for k, v in state_dict.items()]):
            state_dict = {
                k.replace("control_model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("control_model.")
            }

        save_file(state_dict, os.path.splitext(filepath)[0] + ".safetensors")
        if config.wipe_older_ckpts:
            for f in os.listdir(os.path.dirname(filepath)):
                if f.endswith(".ckpt") and f != os.path.basename(filepath):
                    os.remove(os.path.join(os.path.dirname(filepath), f))


def prepare_model_for_training():
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.

    base_model_path, base_model_config = create_controlnet_model(
        sd_version=config.sd_version
    )

    model = create_model(base_model_config).cpu()
    model.load_state_dict(load_state_dict(base_model_path, location="cpu"))
    model.learning_rate = config.learning_rate
    model.sd_locked = config.sd_locked
    model.only_mid_control = config.only_mid_control
    return model


def get_latest_ckpt():
    ckpt_list = os.listdir(config.output_dir)
    ckpt_list = [x for x in ckpt_list if x.endswith(".ckpt")]
    if len(ckpt_list) > 0:
        ckpt_list = sorted(
            ckpt_list,
            key=lambda x: os.path.getmtime(os.path.join(config.output_dir, x)),
            reverse=True,
        )
        found_ckpt = os.path.join(config.output_dir, ckpt_list[0])
    else:
        found_ckpt = None

    return found_ckpt


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ""
    p = name[: len(parent_name)]
    if p != parent_name:
        return False, ""
    return True, name[len(parent_name) :]


def create_controlnet_model(sd_version="2.1"):
    script_dir_path = os.path.dirname(os.path.realpath(__file__))
    models_folder_path = os.path.join(script_dir_path, "models")
    output_path = (
        os.path.join(models_folder_path, "control_sd21_ini.ckpt")
        if sd_version == "2.1"
        else os.path.join(models_folder_path, "control_v15_ini.ckpt")
    )
    config_file = (
        "./models/cldm_v21.yaml" if sd_version == "2.1" else "./models/cldm_v15.yaml"
    )
    if not os.path.exists(output_path):
        model = create_model(config_path=config_file)
        sd_path = (
            "./models/v2-1_512-ema-pruned.ckpt"
            if sd_version == "2.1"
            else "./models/v1-5-pruned.ckpt"
        )
        if not os.path.exists(sd_path):
            '''url = (
                "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt"
                if sd_version == "2.1"
                else "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt"
            )'''
            url = (
                "https://huggingface.co/Manojb/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt"
                if sd_version == "2.1"
                else "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt"
            )
            print("Downloading pretrained model...")
            with tqdm(
                unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
            ) as t:
                urllib.request.urlretrieve(
                    url,
                    filename=sd_path,
                    reporthook=lambda b, bsize, tsize: t.update(bsize),
                )
        else:
            print("Pretrained model already exists, skipping download...")

        pretrained_weights = torch.load(sd_path, weights_only=False)
        if "state_dict" in pretrained_weights:
            pretrained_weights = pretrained_weights["state_dict"]

        scratch_dict = model.state_dict()

        target_dict = {}
        for k in scratch_dict.keys():
            is_control, name = get_node_name(k, "control_")
            if is_control:
                copy_k = "model.diffusion_" + name
            else:
                copy_k = k
            if copy_k in pretrained_weights:
                target_dict[k] = pretrained_weights[copy_k].clone()
            else:
                target_dict[k] = scratch_dict[k].clone()
        #        print(f'These weights are newly added: {k}')

        model.load_state_dict(target_dict, strict=True)
        torch.save(model.state_dict(), output_path)
        os.remove(sd_path)

    return output_path, config_file
