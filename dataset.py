import json
import cv2
import numpy as np
from config import config
import os
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open(config.dataset_captions_json, "rt", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]

        prompt = item["prompt"]

        source_path = os.path.join(config.dataset_conditioning_folder, source_filename)
        target_path = os.path.join(config.dataset_target_folder, target_filename)

        source = cv2.imread(source_path)
        target = cv2.imread(target_path)

        # resize source image to config.resolution
        source = cv2.resize(source, (config.resolution, config.resolution))

        # resize target image to config.resolution
        target = cv2.resize(target, (config.resolution, config.resolution))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
