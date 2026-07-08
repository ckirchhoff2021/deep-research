import os
from openai import OpenAI
from tools.image import image_to_base64
from tools.logger import setup_logger

logger = setup_logger("skillevolve.chat")


class ChatModel:
    """
    Chat Completion 模型包装
    支持图像和文本推理，可作为 Target 模型（冻结）或 Optimizer 模型
    """
    def __init__(self, base_url, api_key, model_name, enable_thinking=False):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        
    def infer_with_image(self, image_input, skill_prompt, user_prompt=None, temperature=0.1):
        """对单张图像进行推理"""
        user_text = user_prompt or "请按照skill要求分析输入图像"
        messages = [
            {"role": "system", "content": f"<skill>{skill_prompt}</skill>"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_to_base64(image_input)}},
                {"type": "text", "text": user_text}
            ]}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        )
        return {"result": response.choices[0].message.content}
        
    def infer_with_text(self, text_input, skill_prompt="", temperature=0.7):
        """对文本进行推理"""
        logger.info(f"[infer_with_text] 输入 text_input: \n{text_input}")
        logger.info(f"[infer_with_text] 输入 skill_prompt: \n{skill_prompt}")
        # logger.info(f"[infer_with_text] 输入 temperature: \n{temperature}")
        
        messages = []
        if skill_prompt:
            messages.append({"role": "system", "content": f"<skill>{skill_prompt}</skill>"})
        messages.append({"role": "user", "content": text_input})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        )
        logger.info(f"[infer_with_text] 输出 result: \n{response.choices[0].message.content}")
        
        return {"result": response.choices[0].message.content}
