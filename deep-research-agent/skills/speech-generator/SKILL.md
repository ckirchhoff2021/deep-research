---
name: speech-generator
description: 专业语音生成技能，支持语音克隆、跨语言合成、指令遵循生成三种能力，基于CosyVoice实现。当用户要求生成语音、语音复刻、指定音色生成音频、跨语言/方言语音合成、按指令（如指定语言、语气、语速）生成语音时必须使用此技能，即使没有明确提及语音生成也要触发。
---

# speech-generator 语音生成技能
## 核心能力
1. **语音克隆（voice_clone）**：根据参考音频复刻音色，生成相同音色的语音
2. **跨语言合成（cross_lingual_gen）**：生成指定方言/语言的语音，支持中文方言（四川话、粤语、东北话等）和多语种
3. **指令遵循生成（instruct_gen）**：根据自然语言指令生成语音，支持指定语言、语气、语速等要求

## 使用前提
- 已部署CosyVoice项目，路径：/home/chenxiang.101/workspace/CosyVoice
- 参考音频可提供自定义路径，默认使用自带参考音频：./asset/zero_shot_prompt.wav

## 使用步骤
1. 确定用户需要的生成类型：语音克隆/跨语言合成/指令遵循生成
2. 提取用户输入的合成文本、参考音频路径（可选）、生成指令（可选）
3. 调用封装脚本执行合成
4. 返回生成音频的下载链接给用户

## 输入参数
| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| task_type | string | 是 | 生成类型：`voice_clone`/`cross_lingual_gen`/`instruct_gen` |
| tts_text | string | 二选一 | 需要合成的文本内容（适合短文本，长度建议<2000字） |
| tts_file | string | 二选一 | 包含待合成内容的文本文件路径（适合长文本，避免超时/参数过长问题） |
| prompt_wav | string | 否 | 参考音频路径，默认：./asset/zero_shot_prompt.wav |
| instruct_prompt | string | 否 | 生成指令，仅指令遵循生成时必填，例如"用粤语生成"、"用温柔的女声生成" |
| background | boolean | 否 | 是否后台执行（默认为false），长文本生成时开启可避免超时问题，生成完成后自动保存到输出路径 |
| output_file | string | 否 | 自定义输出文件路径，仅在background模式下使用 |

## 输出格式
```
✅ 语音生成成功，生成音频已可下载：
[音频名称.wav](sandbox:生成文件绝对路径)
* 合成文本：[你的合成文本]
* 任务类型：[语音克隆/跨语言合成/指令遵循生成]
* [可选]参考音频：[参考音频路径]
* [可选]生成指令：[用户指定的生成指令]
```

## 示例
### 示例1：语音克隆
用户请求："用我提供的这个音频的音色，生成一段'欢迎来到人工智能世界'的语音"
执行命令：
```
.venv/bin/python [YOUR_SKILLS_DIR]/speech-generator/scripts/generator.py \
  --task_type voice_clone \
  --tts_text "欢迎来到人工智能世界" \
  --prompt_wav "/user/upload/reference.wav"
```
### 示例2：跨语言合成
用户请求："用四川话念一段绕口令：八百标兵奔北坡"
执行命令：
```
.venv/bin/python [YOUR_SKILLS_DIR]/speech-generator/scripts/generator.py \
  --task_type cross_lingual_gen \
  --tts_text "[四川话]八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮"
```
### 示例3：指令遵循生成
用户请求："用粤语生成这句话：今天真是个好日子"
执行命令：
```
.venv/bin/python [YOUR_SKILLS_DIR]/speech-generator/scripts/generator.py \
  --task_type instruct_gen \
  --tts_text "今天真是个好日子" \
  --instruct_prompt "用粤语生成"
```

### 示例4：长文本/散文生成（避免超时）
用户请求："把这篇1000字的散文合成磁性男声语音"
执行步骤：
1. 先将文本保存到本地文件：/tmp/speech_content.txt
2. 后台执行生成，避免超时：
```
.venv/bin/python [YOUR_SKILLS_DIR]/speech-generator/scripts/generator.py \
  --task_type instruct_gen \
  --tts_file "/tmp/speech_content.txt" \
  --instruct_prompt "用有磁性的男声生成，语速适中" \
  --background true \
  --output_file "prose_recitation.wav"
```
生成完成后可以直接返回输出文件路径给用户。

## 最佳实践
- 文本长度超过2000字时，必须使用`--tts_file`参数传入文本，不要直接用`--tts_text`避免命令行参数过长报错
- 文本长度超过500字时，建议开启`--background true`后台执行，避免生成时间过长导致超时
- 后台执行时生成日志保存在CosyVoice目录下的generation.log文件，可用于排查问题
