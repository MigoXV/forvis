from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision
from fairseq.data import FairseqDataset

from forvis.datasets.minio_config import MinIOConfig
from forvis.datasets.minio_driver import MinIOImageDriver


class OrganMinIODataset(FairseqDataset):

    def __init__(
        self, organ_dict: Dict, data_df: pd.DataFrame, minio_config: MinIOConfig
    ):

        super().__init__(organ_dict=organ_dict, data_df=data_df)
        self.organ_dict = organ_dict
        self.data_df = data_df
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )
        self.minio_driver = MinIOImageDriver(minio_config=minio_config)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        # image = imageio.imread(row["image"])
        image = self.minio_driver.get_image(Path(row["image"]))
        image = self.transform(image)
        organ = self.organ_dict[row["organ"]]
        return image, organ

    def size(self, indice):
        return 1

    def num_tokens(self, indice):
        return 1

    def collater(self, samples):
        images, organs = zip(*samples)
        organs = torch.tensor(organs)
        images = torch.stack(images)
        return images, organs
