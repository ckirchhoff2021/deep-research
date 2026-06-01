import os
import json
import subprocess
from dotenv import load_dotenv
import argparse
import sys

# 加载 .env 文件
load_dotenv()

# 获取当前文件目录，这样我们能找到 infer.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INFER_SCRIPT = os.path.join(CURRENT_DIR, 'analyze.py')

def inference_by_subprocess(prompt, image_file):
    '''
    使用 subprocess 调用 infer.py 进行推理
    '''
    cmd = [
        sys.executable,
        INFER_SCRIPT,
        "--prompt", prompt,
        "--image_file", image_file,
        "--output-json"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # 解析输出，找到 JSON 部分
        output_lines = result.stdout.strip().split('\n')
        # 我们从最后一行开始查找，因为 JSON 可能被打印在最后
        for line in reversed(output_lines):
            line = line.strip()
            if line.startswith('{'):
                try:
                    data = json.loads(line)
                    if 'output_text' in data:
                        return data['output_text'], data.get('reason_text', '')
                except:
                    continue
        
        # 如果找不到 JSON，尝试直接用最后一行
        if output_lines:
            return output_lines[-1].strip()
        
    except subprocess.CalledProcessError as e:
        print(f"Error calling infer.py: {e}")
        print(f"Stderr: {e.stderr}")
    
    return None


def inference(prompt, test_file):
    '''
    test_file: json file path to evaluate, format is {img_path: label}
    '''
    outputs = list()
    datas = json.load(open(test_file, 'r'))
    for img_path in datas.keys():
        print(f"\nProcessing: {img_path}")
        preds = inference_by_subprocess(prompt, img_path)
        if isinstance(preds, tuple):
            pred, reason = preds
        else:
            pred = preds
            reason = ""
        
        if pred is None:
            print(f"  Warning: Failed to get prediction for {img_path}")
            pred = ""
            
        import re
        match = re.search(r'<result>(.*?)</result>', pred, re.DOTALL)
        result = match.group(1).strip() if match else pred
        outputs.append({
            'img_path': img_path,
            'reason': reason,
            'pred': result,
            'label': datas[img_path],
            'full_pred': pred
        })

    return outputs


def calculate_metrics(outputs):
    '''
    outputs: list of dict, each dict is {'img_path': img_path, 'pred': pred, 'label': label}
    return accuracy
    '''
    results = {}
    labels = set([x['label'] for x in outputs])
    for label in labels:
        tp = fp = tn = fn = 0
        for metric in outputs:
            pred = metric["pred"]
            gt = metric["label"]
            if label == gt: 
                if pred == gt:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == label:
                    fp += 1
                else:
                    tn += 1
        
        accuracy = (tp + tn) / (tp + fp + tn + fn) 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        results[label] = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
        
    corrects = [int(x['label'] == x['pred']) for x in outputs]
    accuracy = sum(corrects) / len(corrects)
    results['global_accuracy'] = round(accuracy, 4)
    return results


def save_outputs(outputs):
    import time
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    output_file = f"evals-{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f'Successfully save outputs to {output_file}')
    

def main() -> None:
    parser = argparse.ArgumentParser(description="chart analysis")
    parser.add_argument("--prompt_file", type=str, help="prompt file path to evaluate")
    parser.add_argument("--test_file", type=str, help="sample file path to evaluate")
    args = parser.parse_args()
    
    prompt = open(args.prompt_file, 'r').read()
    outputs = inference(prompt, args.test_file)
    results = calculate_metrics(outputs)
    save_outputs({
        'metrics': results,
        'predicts': outputs,
    })    
    print(results)



if __name__ == '__main__':
    main()
