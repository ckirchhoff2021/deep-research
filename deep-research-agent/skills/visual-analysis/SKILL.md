---
name: visual-analysis
description: 视觉多模态分析技能，支持三大核心能力：1. 使用用户提供的模型或者内置模型，对输入的图像进行多模态理解分析，支持开启/关闭thinking模式输出完整推理过程；2. 根据用户提供的分类知识和评测集，评测内置模型或者用户提供的模型在评测集上的表现；3. 根据用户输入的评测集和分类prompt，使用内置模型评测prompt在评测集上的表现。当用户要求分析图像、评估新部署的模型效果，评测视觉分类prompt效果时必须使用本技能。
---

# Visual Analysis Skill
专业的视觉多模态分析与分类效果评测技能，支持完整的分析-评测-优化全链路能力。

## 核心功能与使用流程
### 功能1：图像多模态分析
**触发场景**：用户上传图像并要求描述内容、分析图表趋势、识别图像特征时使用
**使用方法**：
```bash
.venv/bin/python [YOUR_SKILLS_DIR]/visual-analysis/scripts/analyze.py  \
  --base_url <模型URL> \
  --api_key <API密钥> \
  --model_name <模型名称> \
  --temperature <温度参数> \
  --image_file <输入图像绝对路径> \
  --user_prompt <分析指令文本或指令文件路径> \
  --system_prompt <系统提示文本> \
  --api_type <API类型> \
  --reasoner <推理深度> \
  [--output-json] \  
  [--thinking] # 可选参数，开启后输出完整推理过程
```
**参数说明**：
| 参数 | 说明 | 示例 |
|------|------|------|
| `{base_url}` | 模型 API 地址 | `https://api.openai.com/v1` |
| `{api_key}` | 模型API KEY | `Empty` |
| `{model_name}` | 模型名称 | `Qwen3.5-0.8B` |
| `{temperature}` | 温度参数 | `0.2` |
| `{image_file}` | 输入图像绝对路径 | `/home/workspace/offset_236.png` |
| `{user_prompt}` | 分析指令文本或指令文件路径 | `分析 offset_236.png 这张图表的走势` |
| `{system_prompt}` | 系统提示文本 | `你是一个专业的视觉分析助手，能够根据图像内容进行分类和分析。` |
| `{api_type}` | API类型 | `response/chat-completion`，如果是自部署模型，必须使用`chat-completion`类型，内置模型用`response`类型 |
| `{reasoner}` | 推理深度 | `medium` |
| `{output_json}` | 是否输出 JSON 格式结果 | `True`，store_true|
| `{thinking}` | 是否开启 thinking 模式 | `True`, store_true |

**示例**：
输入：`分析 offset_236.png中红色曲线相比蓝色基准曲线的趋势和异常恢复状态，分析标准参考knowledge.md`

输出：
```
### 分析结果
| 指标 | 结果 |
|------|------|
| 趋势分类 | offset |
| 恢复状态 | no_recover |

#### 分析依据：
红色Current曲线与蓝色Reference曲线全程无交点完全分离，红色始终位于蓝色上方，符合offset类别特征，末尾水位差无缩小趋势，无恢复迹象。
```
---

### 功能2：Prompt和模型在指定评测集上进行精度评测
**触发场景**：用户提供prompt和视觉分类评测集，要求评估模型在评测集上的分类效果
**使用方法**：
```bash
.venv/bin/python [YOUR_SKILLS_DIR]/visual-analysis/scripts/evaluator.py \
  --base_url <模型URL> \
  --api_key <API密钥> \
  --model_name <模型名称> \
  --system_prompt <系统提示文本> \
  --user_prompt <分析指令文本或指令文件路径> \
  --temperature <温度参数> \
  --api_type <API类型> \
  --output_file <输出JSON绝对路径> \
  --test_file <评测集JSON绝对路径> \
  --result_pattern <从输出文本中抽取结果的正则表达式> \
  [--thinking] # 可选参数，开启后输出完整推理过程
```
**参数说明**：
| 参数 | 说明 | 示例 |
|------|------|------|
| `{base_url}` | 模型 API 地址 | `https://api.openai.com/v1` |
| `{api_key}` | 模型API KEY | `Empty` |
| `{model_name}` | 模型名称 | `Qwen3.5-0.8B` |
| `{temperature}` | 温度参数 | `0.2` |
| `{user_prompt}` | 分析指令文本或指令文件路径 | `分析 offset_236.png 这张图表的走势` |
| `{system_prompt}` | 系统提示文本 | `你是一个专业的视觉分析助手，能够根据图像内容进行分类和分析。` |
| `{api_type}` | API类型 | `response/chat-completion`，如果是自部署模型，必须使用`chat-completion`类型，内置模型用`response`类型|
| `{thinking}` | 是否开启 thinking 模式 | `True`, store_true |
| `{result_pattern}` | 从输出文本中抽取结果的正则表达式，一般从user_prompt或者system_prompt中输出各种中提取到 | `<result>(.*?)</result>` |
| `{test_file}` | 评测集JSON绝对路径 | `/home/workspace/test.json` |
| `{output_file}` | 输出JSON绝对路径 | `/outputs/evals-20260601112410.json` |


**输出要求**：
必须包含以下内容：
1. 全局准确率指标
2. 分类别精度详情表（准确率、精确率、召回率、表现说明）
3. 核心问题总结
4. 评测结果文件保存路径

**示例输出参考**：
| 指标 | 数值 |
|------|------|
| 全局准确率 | 83.33% |

| 类别 | 准确率 | 精确率 | 召回率 | 表现说明 |
|------|--------|--------|--------|----------|
| offset | 100% | 100% | 100% | 表现完美 |
| normal | 100% | 100% | 100% | 误判问题完全解决 |
| fluctuate | 88.89% | 100% | 33.33% | 仍存在漏判问题 |

核心问题：fluctuate类别召回率较低，仍需优化判据边界。
结果保存路径：`/outputs/evals-20260601112410.json`

---

### 功能3：视觉分类Prompt优化建议
**触发场景**：用户需要评估指定视觉分类prompt在指定评测集上的分类效果，要求优化prompt以提升分类精度
**优化流程**：
1. 检查上下文是否有评测结果，若没有结果，则使用功能2生成评测结果文件
2. 审视输入的prompt，从以下维度给出优化建议：
   - 分类标准是否清晰 
   - 类别之间的判定是否有冲突
   - 类别判定优先级是否清晰合理
   - 从全局判断分类标准是否逻辑严谨
3. 审视评测结果，定位错误case的共性规律，给出优化建议
4. 生成优化后的prompt版本，并给出精度提升预期
5. 使用功能2再次自动运行评测验证优化效果，输出前后精度对比表

**输出要求**：
必须包含：
1. 原始prompt的核心缺陷总结
2. 具体优化点说明
3. 优化前后精度对比表
4. 优化后的prompt完整内容和保存路径

---

## 依赖说明
- 支持的图像格式：PNG、JPG、JPEG
- 支持的评测集格式：标准JSON格式，包含图像路径、真实标签、分类标注信息， 格式为：
```json
{
    "{image_path}": "{label}"
}
```
