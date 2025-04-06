import io
import logging
from collections.abc import Callable
from typing import List

import imageio
import numpy as np
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from forvis.infer.resnet18.infer import Inferencer
from forvis.web.infer import get_inferencer, get_llm_inferencer

logger = logging.getLogger(__name__)


app = FastAPI()
# 允许所有源的配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,  # 是否允许发送 cookie
    allow_methods=["*"],  # 允许所有 HTTP 方法（GET, POST, PUT, DELETE, OPTIONS 等）
    allow_headers=["*"],  # 允许所有 HTTP 头
)

# # 静态文件目录设置
# app.mount("/assets", StaticFiles(directory="web/assets", html=False), name="assets")
app.mount("/gui", StaticFiles(directory="gui/", html=False), name="gui")
# app.mount("/", StaticFiles(directory="web/", html=True), name="web")

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def frontend(full_path: str):
    with open("gui/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# 新增路由 /organ，当该接口传入一张图片时，将图片转为 numpy 数组并打印其形状
@app.post("/organ")
async def organ_inference(
    file: UploadFile = File(...),
    inferencer: Inferencer = Depends(get_inferencer),
    llm_inferencer: Callable = Depends(get_llm_inferencer),
):
    # 读取上传的图片数据
    contents = await file.read()
    # 使用 PIL 打开图片
    image = imageio.imread(io.BytesIO(contents))
    organ, prob = inferencer.infer(image)
    # 使用 LLM 进行推理
    prompt = f"这是一张{organ}的法医鉴定切片，你需要描述这张图片的组织学特征和可能的病理学背景。"
    report = llm_inferencer(prompt=prompt, image=image)
    return {
        "filename": file.filename,
        "organ": organ,
        "probability": prob,
        "report": report,
    }
