import io
import base64
from io import BytesIO
from PIL import Image

    
def image_to_base64(image_input):
    """将图像转为 base64"""
    if isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        image = Image.open(image_input).convert('RGB')
        
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'