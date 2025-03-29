import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import sklearn
import sklearn.model_selection
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from forvis.datasets.minio_config import MinIOConfig
from forvis.datasets.organ_minio import OrganMinIODataset


@dataclass
class OrganMinIOConfig(FairseqDataclass):
    organ_dict: str = field(
        default=os.getenv("ORGAN_DICT", ""),
        metadata={"help": "path to organ dictionary file"},
    )
    trian_tsv: str = field(
        default=os.getenv("ORGAN_TRAIN_CSV", ""),
        metadata={"help": "path to train organ classification tsv file"},
    )
    val_tsv: Optional[str] = field(
        default=os.getenv("ORGAN_VAL_CSV"),
        metadata={"help": "path to val organ classification tsv file"},
    )
    val_size: Optional[float] = field(
        default=0.1,
        metadata={"help": "size of validation set"},
    )
    minio_config: MinIOConfig = field(
        default=MinIOConfig(),
        metadata={"help": "MinIO configuration"},
    )


@register_task("organ_minio", dataclass=OrganMinIOConfig)
class OrganMinioClassficationTask(FairseqTask):

    def __init__(self, config: OrganMinIOConfig):
        super().__init__(config)
        self.config = config
        self.organ_dict = json.load(open(config.organ_dict))
        self.train_df = pd.read_csv(config.trian_tsv, sep="\t")
        if config.val_tsv:
            self.val_df = pd.read_csv(config.val_tsv, sep="\t")
        else:
            self.train_df, self.val_df = sklearn.model_selection.train_test_split(
                self.train_df, test_size=config.val_size
            )

    def load_dataset(self, split, **kwargs):
        if split == "train":
            self.datasets[split] = OrganMinIODataset(
                self.organ_dict, self.train_df, self.config.minio_config
            )
        elif split == "valid":
            self.datasets[split] = OrganMinIODataset(
                self.organ_dict, self.val_df, self.config.minio_config
            )
        else:
            raise KeyError(f"Invalid split: {split}")

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None
