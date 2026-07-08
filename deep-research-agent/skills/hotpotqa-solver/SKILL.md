---
name: hotpotqa-solver
description: 解决HotpotQA多跳问答任务，采用迭代检索-推理流水线，自动计算检索命中率和QA准确率，支持全数据集评估和详细结果分析。当用户需要解决HotpotQA问题、评估多跳QA性能、测试检索增强生成(RAG)在多跳任务上的效果、或分析大模型多步推理能力时使用本技能。
---

# HotpotQA多跳问答求解器

本技能用于解决HotpotQA多跳问答任务，采用自适应迭代检索-推理架构，支持自定义模型配置、全量数据集评估和核心指标自动计算。

## 核心能力
- **迭代检索-推理流水线**：模型自主判断证据充足性，不足时自动生成检索查询，最多支持6步多跳推理
- **向量检索模块**：基于FAISS内积相似度的Top1文档检索，自动从干扰文档池中定位相关证据，避免重复检索
- **双API兼容**：支持OpenAI格式的Chat Completion API和Responses API两种调用方式，适配不同部署的模型
- **双指标自动计算**：
  - **QA准确率**：通过LLM自动评判预测答案与标准答案的一致性
  - **检索Hit@1命中率**：统计检索步骤命中黄金支持文档的比例
- **完整推理链记录**：保存每一步的检索查询、检索结果和模型输出，便于调试分析
- **环境变量支持**：可通过.env文件配置默认API参数，简化调用

## 目录结构
```
hotpotqa-solver/
├── SKILL.md                    # 本说明文档
├── assets/
│   ├── instruction.md          # 多跳推理系统提示词
│   └── judge.md                # 答案正确性评判提示词
├── scripts/
│   ├── solver.py               # 主运行脚本
│   ├── models/                 # 模型封装
│   │   ├── chat.py             # Chat Completion API模型实现
│   │   ├── response.py         # Responses API模型实现
│   │   └── embed.py            # 嵌入模型封装
│   ├── rollouts/
│   │   └── hotpotqa.py         # HotpotQA推理核心逻辑
│   └── tools/                  # 工具模块
│       ├── retrieve.py         # FAISS向量检索实现
│       ├── common.py           # 结果提取等通用函数
│       ├── logger.py           # 日志配置
│       └── image.py            # 图像处理工具（预留）
└── outputs/                    # 日志和结果输出目录（自动创建）
```

## 使用流程

### 步骤1：准备数据集
HotpotQA数据集需要提前准备，支持两种方式：
1. 本地已下载：直接指定本地数据集路径即可
2. 未下载：从HuggingFace下载官方数据集：[hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa)，下载后指定本地路径

> 说明：脚本默认使用`distractor`分割，每个问题包含10篇文档（2篇黄金支持文档+8篇干扰文档），是标准的多跳推理测试场景。

### 步骤2：配置参数
#### 方式一：命令行参数
运行脚本时通过命令行指定参数：

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `--dataset_path` | str | 是 | - | HotpotQA数据集本地路径 |
| `--api_base` | str | 否 | 用户提供或者使用脚本中的默认值 | 大模型API服务地址 |
| `--api_key` | str | 否 | 用户提供或者使用脚本中的默认值 | 大模型API密钥 |
| `--model_name` | str | 否 | 用户提供或者使用脚本中的默认值 | 大模型名称 |
| `--api_type` | str | 否 | `chat_completion` | API类型，可选值：<br>`chat_completion`：适用于Qwen等大部分开源模型<br>`response_completion`：适用于Ark平台部署的模型 |
| `--embedding_api_base` | str | 否 | 用户提供或者使用脚本中的默认值 | 嵌入模型API地址（默认与大模型API地址相同） |
| `--embedding_api_key` | str | 否 | 用户提供或者使用脚本中的默认值 | 嵌入模型API密钥（默认与大模型API密钥相同） |
| `--embedding_model` | str | 否 | 用户提供或者使用脚本中的默认值 | 嵌入模型名称，推荐使用英文语义相似度模型如`BAAI/bge-base-en-v1.5` |
| `--num_samples` | int | 否 | 100 | 评估样本数量，设为0使用全部验证集 |
| `--max_steps` | int | 否 | 6 | 每个问题的最大检索-推理步数 |
| `--enable_thinking` | flag | 否 | False | 是否启用模型思考模式（仅支持thinking能力的模型可用） |
| `--log_dir` | str | 否 | `[YOUR_SKILLS_DIR]/hotpotqa-solver/outputs/logs/hotpot.log` | 日志文件输出路径 |

当用户提供命令行参数后，在`scripts/.env`文件中写入参数，避免下次运行重复输入。

#### 方式二：环境变量配置
在`scripts/`目录下创建`.env`文件，配置默认参数，避免每次运行重复输入：
```env
# 大模型配置
API_KEY=your_api_key
API_URL=https://your-llm-endpoint/v1
MODEL_NAME=your_model_name

# 嵌入模型配置
EMBED_BASE_URL=https://your-embedding-endpoint/v1
EMBED_API_KEY=your_embedding_api_key
EMBED_MODEL=your_embed_model
```

### 步骤3：运行求解器
```bash
# 进入技能脚本目录
cd [YOUR_SKILLS_DIR]/hotpotqa-solver/scripts

# 运行示例（命令行指定参数）
python solver.py \
  --dataset_path /path/to/your/hotpot_qa \
  --api_base "https://your-llm-endpoint/v1" \
  --api_key "your_api_key" \
  --model_name "your_model_name" \
  --embedding_model "your_embed_model" \
  --embedding_api_base "https://your-embedding-endpoint/v1" \
  --embedding_api_key "your_embedding_api_key" \
  --num_samples 100 \
  --max_steps 6
```

### 步骤4：结果查看
#### 控制台输出
运行完成后会直接打印核心指标：
```
  [Rollout] Processed 10/100
  [Rollout] Processed 20/100
  ...
Accuracy: 0.6200
Retrieved Accuracy hit@1: 0.7500
```
- `Accuracy`：QA准确率，即回答正确的问题占总样本数的比例
- `Retrieved Accuracy hit@1`：检索命中率，即所有检索步骤中命中黄金支持文档的比例

#### 日志文件
详细的推理过程和API交互日志会保存到`--log_dir`指定的路径（默认`outputs/logs/hotpot.log`），包含：
- 每个问题的完整推理链
- 每一步的模型输入输出内容
- 检索查询词和检索到的文档内容
- 运行过程中的错误信息和异常栈

## 提示词自定义
核心提示词位于`assets/`目录，可根据需要自行调整优化：
1. `instruction.md`：多跳推理系统提示词，指导模型何时需要检索、如何生成检索查询、何时输出最终答案，以及输出格式规范
2. `judge.md`：答案评判提示词，指导模型判断预测答案与标准答案是否语义一致，要求只能输出`Yes`或`No`，不能包含其他内容

## 注意事项
1. 嵌入模型需要支持良好的英文语义相似度计算能力
2. 大模型需要具备较强的多步推理能力和指令遵循能力，推荐使用7B参数以上的指令微调模型
3. 首次运行如果传入HuggingFace数据集路径，会自动下载数据集，请确保网络通畅
4. 样本数量较大时运行时间较长，建议先使用小样本（如10-20个）测试流程正确性
5. 模型必须严格遵循输出格式要求：检索时使用`<retrieve>查询</retrieve>`，输出最终答案时使用`<result>答案</result>`，否则会被判定为格式错误
6. HotpotQA为英文数据集，问题、文档和答案均为英文，无需额外翻译
7. 日志目录会自动创建，无需手动新建
