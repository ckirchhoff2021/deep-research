from openai import OpenAI
from langchain_core.tools import tool
from PIL import Image
import base64
from io import BytesIO


def convert_image_to_base64(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert('RGB')
        output_buffer = BytesIO()
        fmt = "jpeg"
        image.save(output_buffer, format=fmt)
        
        byte_data = output_buffer.getvalue()
        b64_str = base64.b64encode(byte_data).decode('utf-8')
        return f'data:image/{fmt};base64,' + b64_str
    except:
        raise ValueError(f"Failed to convert image {image_path} to base64")


@tool
def openai_chat_completion(
    base_url: str,
    api_key: str,
    model_name: str,
    user_prompt: str,
    local_image_path: str = '',
    system_prompt: str = '',
    history: list = [],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    top_p: float = 0.5,
    thinking: bool = False
):
    """
    This tool implements the standard OpenAI Chat Completions interface for invoking large language models
    and is commonly used for inference with self-deployed models.
    
    Args:
        base_url: OpenAI API base URL (e.g., "https://api.openai.com/v1")
        api_key: OpenAI API key
        model_name: Name of the model to use (e.g., "gpt-3.5-turbo")
        system_prompt: System prompt for the model (e.g., "You are a helpful assistant")
        user_prompt: User prompt for the model (e.g., "Hello")
        local_image_path: Local image file path (default "")
        history: List of previous messages (e.g., [{"role": "user", "content": "Hello"}])
        temperature: Temperature for the model (default 0.7)
        max_tokens: Maximum number of tokens to generate (default 1024)
        top_p: Top-p parameter for the model (default 0.5)
        thinking: Whether to enable thinking mode (default False, for Qwen models)
    """
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = []
    if len(history) > 0:
        messages.extend(history)
        
    if len(system_prompt) > 0:
        messages.append(
            {"role": "system", "content": f"{system_prompt}"}
        )
    
    if len(local_image_path) > 0:
        image_b64 = convert_image_to_base64(local_image_path)
        image_input = {
            "type": "image_url",
            "image_url": {"url": image_b64}
        }
        text_input = {"type": "text", "text": user_prompt}
        messages.append(
            {"role": "user", "content": [image_input, text_input]}
        )
    else:
        messages.append(
            {"role": "user", "content": f"{user_prompt}"}
        )     
       
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body= {"chat_template_kwargs": {"enable_thinking": thinking}}
    )
    return response.choices[0].message.content


@tool
def openai_response(
    base_url: str,
    api_key: str,
    model_name: str,
    user_prompt: str,
    local_image_path: str = '',
    system_prompt: str = '',
    reasoner: str = 'medium',
    history: list = [],
    temperature: float = 0.7,
    max_output_tokens: int = 2048,
    top_p: float = 0.5,
    thinking: bool = False,
):
    """
    This is a standard OpenAI Responses API tool used to call large language models，
    which is generally adopted for models deployed via Ark.
    reference: https://developers.openai.com/api/docs/guides/images-vision
    
    Args:
        base_url: OpenAI API base URL (e.g., "https://api.openai.com/v1")
        api_key: OpenAI API key
        model_name: Name of the model to use (e.g., "gpt-3.5-turbo")
        system_prompt: System prompt for the model (e.g., "You are a helpful assistant")
        user_prompt: User prompt for the model (e.g., "Hello")
        local_image_path: Local image file path (default "")
        history: List of previous messages (e.g., [{"role": "user", "content": "Hello"}])
        temperature: Temperature for the model (default 0.7)
        max_output_tokens: Maximum number of tokens to generate (default 1024)
        top_p: Top-p parameter for the model (default 0.5)
        reasoner: Reasoning effort for the model (default "medium"), can be "minimal", "low", "high"
        thinking: Whether to enable thinking mode (default False, for ark models)
    """
    
    if reasoner not in ['minimal', 'low', 'medium', 'high']:
        raise ValueError(f"Invalid reasoninger: {reasoner}. Must be 'minimal', 'low', or 'high'.")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = []
    if len(history) > 0:
        messages.extend(history)
        
    if len(local_image_path) > 0:
        image_b64 = convert_image_to_base64(local_image_path)
        image_input = {
            "type": "input_image",
            "image_url": image_b64
        }
        text_input = {"type": "input_text", "text": user_prompt}
        messages.append(
            {"role": "user", "content": [text_input, image_input]}
        )
    else:
        messages.append(
            {"role": "user", "content": f"{user_prompt}"}
        )     
    
    thinking_type = "disabled" if not thinking else "enabled"
    response = client.responses.create(
        model=model_name,
        instructions=system_prompt,
        input=messages,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={"thinking": {"type": thinking_type}},
    )
    return response.output_text   # response.output[0].summary[0].text
