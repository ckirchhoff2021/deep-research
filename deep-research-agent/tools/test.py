from .openai_tools import openai_chat_completion, openai_response
from dotenv import load_dotenv
import os

load_dotenv()

def case_openai_chat_completion():
    model_name = "Qwen3.5-0.8B"
    base_url = "http://localhost:8000/v1"
    api_key = "Empty"
    response= openai_chat_completion.invoke(
        {"base_url": base_url, "api_key": api_key, "model_name": model_name, "user_prompt": "你好"}
    )
    print(response)
    

def case_openai_chat_completion_image():
    model_name = "Qwen3.5-0.8B"
    base_url = "http://localhost:8000/v1"
    api_key = "Empty"
    local_image_path = "/home/chenxiang.101/workspace/tmp/badcases/20260605-110614.jpg"
    response= openai_chat_completion.invoke(
        {"base_url": base_url, "api_key": api_key, "model_name": model_name, "user_prompt": "描述一下这张图像", "local_image_path": local_image_path}
    )
    print(response)


def case_openai_response():
    model_name = os.getenv("SPEECH_MODEL")
    base_url = os.getenv("SPEECH_BASE_URL")
    api_key = os.getenv("SPEECH_API_KEY")
    response= openai_response.invoke(
        {"base_url": base_url, "api_key": api_key, "model_name": model_name, "user_prompt": "你好"}
    )
    print(response)
    
    
def case_openai_response_image():
    model_name = os.getenv("SPEECH_MODEL")
    base_url = os.getenv("SPEECH_BASE_URL")
    api_key = os.getenv("SPEECH_API_KEY")
    local_image_path = "/home/chenxiang.101/workspace/tmp/badcases/20260605-110614.jpg"
    response= openai_response.invoke(
        {"base_url": base_url, "api_key": api_key, "model_name": model_name, "user_prompt": "描述一下这张图像", "local_image_path": local_image_path}
    )
    print(response)
    
    
if __name__ == '__main__':
    case_openai_chat_completion()
    case_openai_chat_completion_image()
    case_openai_response()
    case_openai_response_image()
