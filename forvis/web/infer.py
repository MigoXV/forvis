import json
import logging
import os
from functools import partial
from forvis.infer.resnet18.infer import Inferencer
from forvis.infer.resnet18.llm import llm_infer
import openai

logger = logging.getLogger(__name__)

inferencer = None
llm_inferencer = None

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

def get_llm_inferencer():
    global llm_inferencer
    if llm_inferencer is None:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        llm_inferencer = partial(
            llm_infer,
            client=client,
            model=os.getenv("LLM_MODEL", "gpt-4o"),
        )
    logger.info(f"Using LLM {os.getenv('LLM_MODEL')}")
    return llm_inferencer
    