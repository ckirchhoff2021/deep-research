from typing import List, Dict, Any
import re


def extract_result(completion, pattern=r'<result>(.*?)</result>'):
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""