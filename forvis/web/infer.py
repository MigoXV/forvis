import json
import logging
import os

from forvis.infer.resnet18.infer import Inferencer

logger = logging.getLogger(__name__)

inferencer = None


def get_inferencer():
    global inferencer
    if inferencer is None:
        ckpt_path = os.getenv("CKPT_PATH")
        inferencer = Inferencer(
            ckpt_path=ckpt_path,
            device=os.getenv("DEVICE", "cpu"),
            organ_dict=json.load(open(os.getenv("ORGAN_DICT"))),
        )
        logger.info(f"Model loaded from {ckpt_path}|Using device: {inferencer.device}")
    return inferencer
