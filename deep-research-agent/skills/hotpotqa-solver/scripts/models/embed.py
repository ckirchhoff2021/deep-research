import os
import numpy as np
from openai import OpenAI


class Embedding(object):
    def __init__(self, base_url, api_key, model_name):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        
    def infer(self, text: list[str]):
        resp = self.client.embeddings.create(
            model= self.model_name,
            input=text,
            encoding_format="float"
        )
        if isinstance(text, list):
            return np.array([item.embedding for item in resp.data]).astype(np.float32)
        else:
            return np.array([resp.data[0].embedding]).astype(np.float32)
