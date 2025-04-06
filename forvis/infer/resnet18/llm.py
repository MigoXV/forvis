import base64
import io
import os

import imageio
import numpy as np
from openai import OpenAI

system_prompt = """
下面是一段可供参考的系统提示词，你可以根据需要进行调整和扩展：

系统提示词：
你是一位具有丰富经验的法医病理学专家，擅长对组织学切片图像进行详细描述和诊断分析。请在回答时严格遵循以下原则：

1. 客观描述与详细分析
   - 依据图像的染色特征、组织结构及形态细节，进行准确、细致的描述。
   - 针对图像中出现的异常区域、病变或囊状结构，提供明确的观察记录及可能的解剖学和病理学背景。

2. 诊断推理与多角度讨论
   - 根据观察到的形态特征，提出可能的诊断方向，并详细说明各项病理指标与鉴别诊断的依据。
   - 对可能的病变进行系统性讨论，同时指出需要进一步检测或其他资料辅助确认的原因。

3. 法医学及临床背景考量
   - 如果图像涉及法医检材，注意区分不同取材部位（例如脑实质与肌肉组织）的组织学特征，避免混淆。
   - 强调诊断意见仅基于图像分析，提醒最终诊断应结合完整的临床、尸检及实验室数据。

4. 专业、严谨与免责声明
   - 使用专业术语和严谨的表述，确保描述和诊断意见准确无误。
   - 在回答中加入免责声明，明确指出图像分析的局限性，并提醒用户最终诊断应依赖于多方面信息和专业判断。

请确保在回答中始终坚持上述要求，提供详细、准确且具有专业深度的分析与意见，确保信息传达客观中立，同时具备实用指导价值。
"""


def llm_infer(
    client: OpenAI,
    prompt: str,
    image: np.ndarray,
    model: str = "gpt-4o",
) -> str:
    """
    使用 LLM 模型进行推理
    :param client: OpenAI 客户端
    :param prompt: 提示词
    :param image: 输入图片，numpy 数组格式
    :param model: 模型名称
    :return: 推理结果
    """
    # 将 numpy 数组转换为 BytesIO 对象
    buffer = io.BytesIO()
    imageio.imwrite(buffer, image, format="png")
    img_bytes = buffer.getvalue()
    # 对图片二进制数据进行 base64 编码，并转换为字符串
    b64_image = base64.b64encode(img_bytes).decode("utf-8")
    # 构造请求消息：包括文本输入和图片输入
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            },
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # client = OpenAI(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     base_url=os.getenv("OPENAI_BASE_URL"),
    # )
    client = OpenAI(
        api_key="sk-sbdmZdLoW8fdCeSY724477Ec8e29444180AcDc62A2C9C779",
        base_url="https://one-api.bltcy.top/v1",
    )
    # 准备提示词和图片路径
    prompt = (
        "这是一张脑的法医鉴定切片，你需要描述这张图片的组织学特征和可能的病理学背景。"
    )
    image_path = "data-bin/test_infer/brain_1-1.jpg"

    # 使用 imageio 读取图片（得到 numpy 数组）
    img_array = imageio.imread(image_path)
    # 使用 OpenAI 客户端进行推理
    result = llm_infer(client, prompt, img_array, model="gpt-4o")
    print(result)
# # 将 numpy 数组写入内存中的 BytesIO 对象（以 PNG 格式保存）
# buffer = io.BytesIO()
# imageio.imwrite(buffer, img_array, format="png")
# img_bytes = buffer.getvalue()

# # 对图片二进制数据进行 base64 编码，并转换为字符串
# b64_image = base64.b64encode(img_bytes).decode("utf-8")

# # 构造请求消息：包括文本输入和图片输入
# response = client.chat.completions.create(
#     model="gpt-4o",  # 或者使用 "gpt-4o-mini"（根据你所使用的模型）
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": prompt},
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{b64_image}"},
#                 },
#             ],
#         }
#     ],
# )

# # 输出模型返回的推理结果
# print(response.choices[0].message.content)
