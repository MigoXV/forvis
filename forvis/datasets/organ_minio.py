from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision
from fairseq.data import FairseqDataset
from forvis.datasets.organ import OrganDataset
from forvis.datasets.minio_config import MinIOConfig
from forvis.datasets.minio_driver import MinIOImageDriver

class OrganMinIODataset(OrganDataset):

    def __init__(self, organ_dict: Dict, data_df: pd.DataFrame,minio_config: MinIOConfig):

        super().__init__(organ_dict=organ_dict, data_df=data_df)
        self.minio_driver = MinIOImageDriver(minio_config=minio_config)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        # image = imageio.imread(row["image"])
        image = self.minio_driver.get_image(Path(row["image"]))
        image = self.transform(image)
        organ = self.organ_dict[row["organ"]]
        return image, organ

