import os
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import base64
from io import BytesIO
import argparse 
import json

load_dotenv()


def parse_prompt(input_prompt):
    '''
    parse prompt from file or string
    '''
    if os.path.exists(input_prompt):
        with open(input_prompt, 'r') as f:
            prompt = f.read()
    else:
        prompt = input_prompt
    return prompt


def convert_image_to_base64(image_path: str, fmt='jpeg') -> str:
    '''
    convert image to base64 string
    '''
    image = Image.open(image_path).convert('RGB')
    output_buffer = BytesIO()
    image.save(output_buffer, format=fmt)
    byte_data = output_buffer.getvalue()
    b64_str = base64.b64encode(byte_data).decode('utf-8')
    return f'data:image/{fmt};base64,' + b64_str


def infer_with_response_api(
    client, model_name, user_prompt, image_path, system_prompt="", thinking=True,
    temperature=0.7, reasoner="medium", output_json=False):
    '''
    infer with Openai Response API, only for ark models
    reasoner: "medium" by default, can be "minimal", "medium", "low", "high"
    thinking: whether to enable thinking mode
    '''
    think = {"type": "enabled"} if thinking else {"type": "disabled"}
    if image_path.startswith('http'):
        image_url = image_path
    else:
        image_url = convert_image_to_base64(image_path)
        
    kwargs = {
        "model": model_name,
        "input": [{ 
            "role": "user",
            "content": [
                { "type": "input_image", "image_url": image_url },
                { "type": "input_text", "text": user_prompt },
            ],
        }],
        "instructions": system_prompt,
        "temperature": temperature,
        "extra_body": {"thinking": think},
    }
    if thinking:
        kwargs["reasoning"] = {"effort": reasoner}
        
    response = client.responses.create(**kwargs)
    output_text = response.output_text
    reason_text = response.output[0].summary[0].text if thinking else ""
        
    if output_json:
        print(json.dumps({"output_text": output_text, "reason_text": reason_text}, ensure_ascii=False))
    else:
        print(f'output_text: {output_text}')
        if len(reason_text) > 0:
            print(f'reason_text: {reason_text}')
    
    
def infer_with_chat_completion(
    client, model_name, user_prompt, image_path, system_prompt="", temperature=0.7, 
    thinking=True, output_json=False):
    '''
    infer with Openai Chat Completion API, only for self-deployed models, like Qwen models
    thinking: whether to enable thinking mode
    '''
    messages = []
    if len(system_prompt) > 0:
        system_prompt = parse_prompt(system_prompt)
        messages.append(
            {"role": "system", "content": f"{system_prompt}"}
        )
    
    user_prompt = parse_prompt(user_prompt)
    if len(image_path) > 0:
        image_b64 = convert_image_to_base64(image_path)
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
        temperature=temperature,
        extra_body= {"chat_template_kwargs": {"enable_thinking": thinking}}
    )
    output_text = response.choices[0].message.content
    if output_json:
        print(json.dumps({"output_text": output_text}, ensure_ascii=False))
    else:
        print(f'output_text: {output_text}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="visual analysis")
    parser.add_argument("--base_url", type=str, default=os.getenv('SPEECH_BASE_URL'), help="base url for openai api")
    parser.add_argument("--api_key", type=str, default=os.getenv('SPEECH_API_KEY'), help="api key for openai api")
    parser.add_argument("--model_name", type=str, default=os.getenv('SPEECH_MODEL'), help="model name for openai api")
    
    parser.add_argument("--user_prompt", type=str, default="Please describe the picture.", help="input prompt or prompt file path for inference")
    parser.add_argument("--system_prompt", type=str, default="", help="system prompt for inference")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature for inference")
    
    parser.add_argument("--image_file", type=str, help="image file or image url to infer")
    parser.add_argument("--thinking", action="store_true", help="Whether to enable thinking mode")
    parser.add_argument("--reasoner", default="medium", type=str, help="reasoner type to use for inference")
    
    parser.add_argument("--api-type", default="response", choices=["response", "chat-completion"], type=str, help="api type to use for inference")
    parser.add_argument("--output-json", action="store_true", help="Output result in JSON format")
    
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    
    if args.api_type == "response":
        infer_with_response_api(
            client, args.model_name, args.user_prompt, args.image_file, args.system_prompt, args.temperature, 
            args.thinking, args.reasoner, args.output_json
        )
    else:
        infer_with_chat_completion(
            client, args.model_name, args.user_prompt, args.image_file, args.system_prompt, args.temperature, 
            args.thinking, args.output_json
        )
    
  
