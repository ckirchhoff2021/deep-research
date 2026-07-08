import os
from openai import OpenAI
from tools.image import image_to_base64
from tools.logger import setup_logger

logger = setup_logger("skillevolve.responses")


class ResponsesModel:
    """
    OpenAI Responses API 模型包装
    支持图像和文本推理，可作为 Target 模型（冻结）或 Optimizer 模型
    """
    def __init__(self, base_url, api_key, model_name, enable_thinking=False):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        
    def infer_with_image(self, image_input, skill_prompt, user_prompt=None, temperature=0.1):
        """对单张图像进行推理"""
        messages = []
        image_b64 = image_to_base64(image_input)
        image_input = {
            "type": "input_image",
            "image_url": image_b64
        }
        text_input = {"type": "input_text", "text": user_prompt or "请按照skill要求分析输入图像"}
        messages.append(
            {"role": "user", "content": [text_input, image_input]}
        )
        
        extra = {"thinking": {"type": "enabled"}} if self.enable_thinking else {"thinking": {"type": "disabled"}}
        response = self.client.responses.create(
            model=self.model_name,
            instructions=f"<skill>{skill_prompt}</skill>" if skill_prompt else "",
            input=messages,
            temperature=temperature,
            reasoning={"effort": "medium"} if self.enable_thinking else None,
            extra_body=extra,
        )
        return {"result": f"{response.output[0].summary[0].text}" if self.enable_thinking else response.output_text,
                "thought": f"{response.output_text}"}
    
    
    def infer_with_text(self, text_input, skill_prompt="", temperature=0.7):
        """对文本进行推理"""
        logger.info(f"[infer_with_text] 输入 text_input: \n{text_input}")
        logger.info(f"[infer_with_text] 输入 skill_prompt: \n{skill_prompt}")
        # logger.info(f"[infer_with_text] 输入 temperature: \n{temperature}")
        
        extra = {"thinking": {"type": "enabled"}} if self.enable_thinking else {"thinking": {"type": "disabled"}}
        response = self.client.responses.create(
            model=self.model_name,
            instructions=f"<skill>{skill_prompt}</skill>" if skill_prompt else "",
            input=text_input,
            temperature=temperature,
            reasoning={"effort": "medium"} if self.enable_thinking else None,
            extra_body=extra,
        )
        
        result = {
            "result": f"{response.output[0].summary[0].text}" if self.enable_thinking else response.output_text,
            "thought": f"{response.output_text}"
        }
        
        logger.info(f"[infer_with_text] 输出 result: \n{result['result']}")
        logger.info(f"[infer_with_text] 输出 thought: \n{result['thought']}")
        
        return result
