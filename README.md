# Deep Research 多模态理解和生成 AI Agent

## 🎯 项目简介
Deep Research 是面向AIGC场景的多模态AI Agent技能集合仓库，提供开箱即用的多模态能力封装，支持快速构建图像/视频/语音/视觉分析类AI应用，所有技能均可独立调用也可组合使用，完美适配Hermes Agent框架和自定义Agent工作流。

## ✨ 核心功能
目前已包含7大核心技能，覆盖全链路AIGC能力：

| 技能名称 | 功能描述 |
|---------|----------|
| 🎨 **image-generator** | AI图像生成与风格转换，支持文生图、图生图、自定义尺寸、多风格切换 |
| 🎬 **video-generator** | AI视频生成与编辑，支持文本生成视频、图像转视频、风格迁移、批量生成 |
| 🎤 **speech-generator** | 智能语音合成，支持20+种中文/英文音色、多情感合成、长文本批量生成、流式输出 |
| 🎧 **speech-analyze** | 语音分析与处理，支持语音转文本、情绪识别、声纹比对、语种检测 |
| 👁️ **visual-analysis** | 图像视觉分析，支持OCR识别、目标检测、内容分类、图像相似度比对、质量评估 |
| 🔍 **semantic-retrieval** | 语义相似度计算与召回，支持文本相似度计算、语义匹配、相似文档召回、top-k结果返回 |
| 🛠️ **skill-creator** | 技能创建与管理工具，快速生成符合规范的新技能模板、自动生成评估用例、打包发布 |

## 📁 目录结构
```
deep-research/
├── deep-research-agent/           # 主Agent项目目录
│   ├── skills/                     # 所有技能集合
│   │   ├── image-generator/        # 图像生成技能
│   │   ├── video-generator/        # 视频生成技能
│   │   ├── speech-generator/       # 语音生成技能
│   │   ├── speech-analyze/         # 语音分析技能
│   │   ├── visual-analysis/        # 视觉分析技能
│   │   ├── semantic-retrieval/     # 语义检索技能
│   │   └── skill-creator/          # 技能创建工具
│   ├── app.py                      # Agent主入口
│   ├── .env.example                # 环境变量配置模板
│   ├── memory/                     # Agent记忆配置
│   ├── prompts/                    # 提示词模板
│   └── tools/                      # 通用工具库
├── README.md                       # 项目说明文档
└── requirements.txt                # 依赖声明
```

### 每个技能的标准目录结构
所有技能均遵循统一的规范结构，方便调用和维护：
```
<skill-name>/
├── SKILL.md              # 技能说明文档（功能、参数、使用示例）
├── assets/               # 静态资源（示例文件、配置、模型等）
├── evals/                # 评估用例与测试数据
├── scripts/              # 可执行脚本与核心功能实现
└── outputs/              # 生成结果输出目录（自动创建）
```

## 🚀 快速开始
### 环境准备
1. 克隆仓库：`git clone https://github.com/ckirchhoff2021/deep-research.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 配置环境变量：复制`.env.example`为`.env`，填写对应API密钥

### 调用技能示例
以生成宫崎骏风格图像为例：
```python
from deep_research_agent.skills.image_generator.scripts.generator import ImageGenerator

generator = ImageGenerator()
image_url = generator.text2image(
    prompt="宫崎骏风格的胖橘猫坐在樱花树下",
    size="2048x1536"
)
print(f"生成的图像地址：{image_url}")
```

或者直接通过命令行调用：
```bash
cd deep-research-agent/skills/image-generator
python scripts/generator.py --prompt "宫崎骏风格的胖橘猫坐在樱花树下" --size "2048x1536"
```

## 📝 分支说明
- `main`：稳定版本分支，经过完整测试可用于生产环境
- `dev`：开发分支，包含最新功能提交
- `merge`：预合并分支，用于多分支合并测试

## 🤝 贡献指南
1. 基于`dev`分支创建功能分支
2. 遵循技能规范开发新功能
3. 提交PR到`dev`分支，审核通过后合并
4. 使用`skill-creator`创建新技能，自动生成符合规范的目录结构

## 📄 许可证
本项目采用MIT许可证，可自由使用和修改。
