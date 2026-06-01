import os
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import base64
from io import BytesIO
import argparse 

load_dotenv()


client = OpenAI(
    base_url=os.getenv('SPEECH_BASE_URL'),
    api_key=os.getenv('SPEECH_API_KEY')
)


def pil_to_base64(image: Image.Image, fmt='jpeg') -> str:
    output_buffer = BytesIO()
    image.save(output_buffer, format=fmt)
    byte_data = output_buffer.getvalue()
    b64_str = base64.b64encode(byte_data).decode('utf-8')
    return f'data:image/{fmt};base64,' + b64_str


def construct_visual_message(prompt, image_path):
    image = Image.open(image_path).convert('RGB')
    image_b64 = pil_to_base64(image)
    
    message = [{
        "role": "user",
        "content": [
            { "type": "input_image", "image_url": image_b64},
            { "type": "input_text", "text": prompt }
        ]
            
    }]
    return message


def infer_by_image_url(prompt, image_url, thinking=True, reasoner="medium", verbose=True):
    '''
    infer by image url
    reasoner: "medium" by default, can be "minimal", "low", "high"
    thinking: whether to enable thinking mode
    '''
    think = {"type": "enabled"} if thinking else {"type": "disabled"}
    kwargs = {
        "model": os.getenv('SPEECH_MODEL'),
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": image_url
                    },
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                ],
            }
        ],
        "extra_body": {"thinking": think},
    }
    if thinking:
        kwargs["reasoning"] = {"effort": reasoner}
    response = client.responses.create(**kwargs)

    output_text = response.output_text
    reason_text = response.output[0].summary[0].text
    if verbose:
        print(f'output_text: {output_text}')
        print(f'reason_text: {reason_text}')
    return output_text, reason_text
  
  

def infer_by_image_file(prompt, image_file, thinking=True, reasoner="medium", verbose=True):
    '''
    infer by image file
    '''
    think = {"type": "enabled"} if thinking else {"type": "disabled"}
    kwargs = {
        "model": os.getenv('SPEECH_MODEL'),
        "input": construct_visual_message(prompt, image_file),
        "extra_body": {"thinking": think},
    }
    if thinking:
        kwargs["reasoning"] = {"effort": reasoner}
    response = client.responses.create(**kwargs)

    output_text = response.output_text
    if thinking:
        reason_text = response.output[0].summary[0].text
    else:
        reason_text = ""
    if verbose:
        print(f'output_text: {output_text}')
        print(f'reason_text: {reason_text}')
    return output_text, reason_text
    
    
def run(prompt, image_file, thinking=True, reasoner="medium", output_json=False):
    if len(prompt) == 0:
        input_prompt = "请描述一下这张图像。"
    else:
        if os.path.exists(prompt):
            input_prompt = open(prompt, 'r').read()
        else:
            input_prompt = prompt
            
    if image_file.startswith("http"):
        output_text, reason_text = infer_by_image_url(input_prompt, image_file, thinking, reasoner, verbose=not output_json)
    else:
        output_text, reason_text = infer_by_image_file(input_prompt, image_file, thinking, reasoner, verbose=not output_json)
    
    if output_json:
        import json
        result = {
            "output_text": output_text,
            "reason_text": reason_text
        }
        print(json.dumps(result, ensure_ascii=False))
    
    return output_text, reason_text
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="chart analysis")
    parser.add_argument("--prompt", type=str, default="", help="input prompt or prompt file path for inference")
    parser.add_argument("--image_file", type=str, help="image file or image url to infer")
    parser.add_argument("--thinking", action="store_true", help="Whether to enable thinking mode")
    parser.add_argument("--reasoner", default="medium", type=str, help="reasoner to use for inference")
    parser.add_argument("--output-json", action="store_true", help="Output result in JSON format")
    
    args = parser.parse_args()
    run(args.prompt, args.image_file, args.thinking, args.reasoner, args.output_json)
  
  
