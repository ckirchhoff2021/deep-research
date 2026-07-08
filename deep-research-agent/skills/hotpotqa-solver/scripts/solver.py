#!/usr/bin/env python3
import os
import argparse
from dotenv import load_dotenv
from tools.logger import setup_logger

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="HotpotQA多跳问答求解器")
    parser.add_argument("--api_base", type=str, default=None, help="用于推理的大模型API地址")
    parser.add_argument("--api_key", type=str, default=None, help="API密钥")
    parser.add_argument("--model_name", type=str, default=None, help="大模型名称")
    parser.add_argument("--api_type", type=str, default="chat_completion", help="API类型，可选值：chat_completion, response_completion")
    
    parser.add_argument("--embedding_api_base", type=str, default=None, help="嵌入模型API地址（默认与api_base相同）")
    parser.add_argument("--embedding_api_key", type=str, default=None, help="嵌入模型API密钥（默认与api_key相同）")
    parser.add_argument("--embedding_model", type=str, default=None, help="嵌入模型名称")
    
    parser.add_argument("--num_samples", type=int, default=100, help="评估样本数量（0表示使用全部验证集）")
    parser.add_argument("--max_steps", type=int, default=6, help="每个问题的最大检索-推理步数")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集路径")
    
    parser.add_argument("--enable_thinking", action="store_true", help="是否启用思考模式")
    
    parser.add_argument("--log_dir", type=str, default="outputs/logs/hotpot.log", help="日志输出目录")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
   
    setup_logger("skillevolve.chat", log_file=args.log_dir, stream=True)
    setup_logger("skillevolve.responses", log_file=args.log_dir, stream=True)
    # setup_logger("skillevolve.embedding", log_file=args.log_dir, stream=True)
    
    from models import ChatModel, ResponsesModel, Embedding
    from rollouts import HotpotQARollout
    
    api_key = args.api_key
    api_base = args.api_base
    model_name = args.model_name
    api_type = args.api_type   
 
    if args.api_key is None:
        api_key = os.getenv("API_KEY")
        api_base = os.getenv("API_URL")
        model_name = os.getenv("MODEL_NAME")
        api_type = "chat_completion"
    
    if api_type == "chat_completion":
        target_model = ChatModel(api_base, api_key, model_name)
    elif api_type == "response_completion":
        target_model = ResponsesModel(api_base, api_key, model_name)
    else:
        raise ValueError(f"Invalid API type: {args.api_type}")

    embedding_api_base = args.embedding_api_base
    embedding_api_key = args.embedding_api_key
    embedding_model = args.embedding_model

    if args.embedding_api_base is None:
        embedding_api_base = os.getenv("EMBED_BASE_URL")
        embedding_model = os.getenv("EMBED_MODEL")
        embedding_api_key = os.getenv("EMBED_API_KEY")
        
    embed_model = Embedding(embedding_api_base, embedding_api_key, embedding_model)
    
    dataset_config = {
        "dataset_info": {
            "dataset_path": args.dataset_path,
            "sub_dir": "distractor",
            "train_sample_num": args.num_samples,
            "validate_sample_num": args.num_samples,
            "max_steps": args.max_steps,
        },
    }
    rollout = HotpotQARollout(target_model, embed_model, dataset_config)
    rollout.rollout_validate()
    

if __name__ == "__main__":
    main()