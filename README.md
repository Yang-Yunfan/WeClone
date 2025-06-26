Skip to content
Navigation Menu
xming521
WeClone

Type / to search
Code
Issues
33
Pull requests
1
Discussions
Actions
Projects
Security
Insights
Owner avatar
WeClone
Public
xming521/WeClone
Go to file
t
Name		
xming521
xming521
Merge pull request #169 from xming521/xming521/issue167
3ab7c7b
 · 
3 days ago
.cursor/rules
🎈 auto fixes by pre-commit hooks
2 weeks ago
.github
fix tg ci
last week
dataset
feat(data): 支持图像识别并优化数据清洗流程
2 weeks ago
examples
更新setting模板
2 weeks ago
tests
0.2.23 运行 weclone-cli server 报错
last week
weclone-audio
🎈 auto fixes by pre-commit hooks
2 weeks ago
weclone
预处理数据报错提示TypeError: can only concatenate str (not "float") to str
3 days ago
.gitignore
预处理数据报错提示TypeError: can only concatenate str (not "float") to str
3 days ago
.pre-commit-config.yaml
注释掉bandit安全检查的配置。
2 weeks ago
LICENSE
Create LICENSE
last month
README.md
更新README.md，添加WeClone平台声明及模型训练效果描述的修改。
last week
ds_config.json
🎈 auto fixes by pre-commit hooks
2 weeks ago
pyproject.toml
将torch依赖版本从2.7.1降级至2.7.0。
last week
settings.template.jsonc
更新版本号至0.2.24，修改torch依赖版本至2.7.1，更新README.md。
last week
Repository files navigation
README
AGPL-3.0 license
download

🚀 One-stop solution for creating your digital avatar from chat history 💡
🚀从聊天记录创造数字分身的一站式解决方案💡
GitHub stars GitHub release WeClone① Twitter Telegram

Featured｜HelloGitHub xming521%2FWeClone | Trendshift Ask DeepWiki

项目主页 ｜ 项目文档 ｜ Windows部署指南 ｜ Linux部署指南【保姆级】

Important

WhatsApp and Telegram chat logs integration for digital avatar creation is coming !
✨核心功能
💫 涵盖打造数字分身的全链路方案，包括聊天数据导出、预处理、模型训练、部署
💬 使用微信聊天记录微调LLM，让大模型有"那味儿"
🔗 绑定到微信、QQ、Telegram、企微、飞书机器人，实现自己的数字分身
🛡️ 隐私信息过滤，本地化微调部署，数据安全可控
📋特性与说明
Important

WeClone 目前未与任何平台合作，未发行任何数字货币。唯一官方网站：weclone.love，谨防仿冒。
Important

WeClone现在支持图片模态数据微调了！并且包含了更全的上下文,记得拉取最新代码并更新依赖。
Important

WeClone仍在快速迭代期，当前效果不代表最终效果。
微调LLM效果很大程度取决于模型大小、聊天数据的数量和质量，理论上模型越大，数据越多，效果越好。
7B模型很容易训练成为大笨蛋，14B模型勉强可以交流，32B及以上的模型效果会更好。
Windows环境未进行严格测试，可以使用WSL作为运行环境。详细教程可点击Windows部署指南查看。
更新日志
[25/06/05]支持图片模态数据微调

硬件要求
项目默认使用Qwen2.5-7B-Instruct模型，LoRA方法对sft阶段微调，大约需要16GB显存。也可以使用LLaMA Factory支持的其他模型和方法。

需要显存的估算值：

方法	精度	7B	14B	30B	70B	xB
Full (bf16 or fp16)	32	120GB	240GB	600GB	1200GB	18xGB
Full (pure_bf16)	16	60GB	120GB	300GB	600GB	8xGB
Freeze/LoRA/GaLore/APOLLO/BAdam	16	16GB	32GB	64GB	160GB	2xGB
QLoRA	8	10GB	20GB	40GB	80GB	xGB
QLoRA	4	6GB	12GB	24GB	48GB	x/2GB
QLoRA	2	4GB	8GB	16GB	24GB	x/4GB
环境搭建
1.cuda安装(已安装可跳过，要求版本12.6及以上)：LLaMA Factory

2.建议使用 uv安装依赖，这是一个非常快速的 Python 环境管理器。安装uv后，您可以使用以下命令创建一个新的Python环境并安装依赖项，注意这不包含音频克隆功能的依赖：

git clone https://github.com/xming521/WeClone.git
cd WeClone
uv venv .venv --python=3.10
source .venv/bin/activate # windows下执行 .venv\Scripts\activate
uv pip install --group main -e . 
3.将配置文件模板复制一份并重命名为settings.jsonc，后续配置修改在此文件进行：

cp settings.template.jsonc settings.jsonc
微调多模态模型时，请使用examples/mllm.template.jsonc作为配置文件。
Note

训练以及推理相关配置统一在文件settings.jsonc

4.使用以下命令测试CUDA环境是否正确配置并可被PyTorch识别，Mac不需要：

python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available());"
5.（可选）安装FlashAttention，加速训练和推理：uv pip install flash-attn --no-build-isolation 版本问题可以使用https://github.com/mjun0812/flash-attention-prebuild-wheels的预编译包安装。

模型下载
国内推荐使用ModelScope下载模型。不建议使用：

git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git
数据准备
请使用PyWxDump提取微信聊天记录（不支持4.0版本微信）。可以先将手机的聊天记录迁移（备份）到电脑，数据量更多一些。下载软件并解密数据库后，点击聊天备份，导出类型为CSV，可以导出多个联系人（不建议使用群聊记录），然后将导出的位于wxdump_tmp/export 的 csv 文件夹放在./dataset目录即可，也就是不同人聊天记录的文件夹一起放在 ./dataset/csv。

图片数据准备
在能进入微信个人文件夹的环境执行，如果没有环境创建环境并安装基础依赖即可（uv pip install -e .），然后执行以下命令，将微信图片数据保存到./dataset/wechat/dat目录下。

python weclone/data/chat_parsers/wechat_parser.py --wechat-data-dir "微信个人文件夹路径 例如 C:\Users\user\Documents\WeChat Files\wxid_d68wiru2zseo22"
之后使用微信图片解密工具解密图片数据,解密后的图片数据保存到dataset/media/images目录下。

数据预处理
项目默认去除了数据中的手机号、身份证号、邮箱、网址。还在settings.jsonc中提供了一个禁用词词库blocked_words，可以自行添加需要过滤的词句（会默认去掉包括禁用词的整句）。
Important

🚨 请一定注意保护个人隐私，不要泄露个人信息！

执行以下命令对数据进行处理，可以根据自己的聊天风格修改settings.jsonc的make_dataset_args。
weclone-cli make-dataset
目前仅支持时间窗口策略，根据single_combine_time_window将单人连续消息通过逗号连接合并为一句，根据qa_match_time_window匹配问答对。
若需训练多模态大模型:通过include_type中添加images启用，并通过image_max_pixels和max_image_num参数控制图片数量和大小，减少显存占用。
若需利用多模态大模型补全数据:在include_type中添加images并配置 vision_api 参数，系统将使用外部多模态模型自动提取图像内容补全数据，最终生成的数据集仍用于训练纯文本语言模型（LLM）。
可以启用clean_dataset中的enable_clean选项，对数据进行清洗，以达到更好效果（多模态数据暂不支持）。* 当前系统支持使用 llm judge 对聊天记录进行打分，提供 vllm 离线推理 和 API 在线推理 两种方式。可通过将 settings.jsonc 文件中的 "online_llm_clear": false 修改为 true 来启用 API 在线推理模式，并配置相应的 base_url、llm_api_key、model_name 等参数。所有兼容 OpenAI 接口的模型均可接入。
在获得 llm 打分分数分布情况 后，可通过设置 accept_score 参数筛选可接受的分数区间，同时可适当降低 train_sft_args 中的 lora_dropout 参数，以提升模型的拟合效果。
配置参数并微调模型
(可选)修改 settings.jsonc 的 model_name_or_path 和 template 选择本地下载好的其他模型。
修改per_device_train_batch_size以及gradient_accumulation_steps来调整显存占用。
可以根据自己数据集的数量和质量修改train_sft_args的num_train_epochs、lora_rank、lora_dropout等参数。
单卡训练
weclone-cli train-sft
多卡环境单卡训练，需要先执行 export CUDA_VISIBLE_DEVICES=0

多卡训练
取消settings.jsonc中deepspeed行代码注释，使用以下命令多卡训练：

uv pip install deepspeed
deepspeed --num_gpus=使用显卡数量 weclone/train/train_sft.py
使用浏览器demo简单推理
可以在这一步测试出合适的temperature、top_p值，修改settings.jsonc的infer_args后，供后续推理时使用。

weclone-cli webchat-demo
使用接口进行推理
weclone-cli server
使用常见聊天问题测试
不包含询问个人信息的问题，仅有日常聊天。测试结果在test_result-my.txt。

weclone-cli server
weclone-cli test-model
🖼️ 微调效果
Tip

QQ群内有部署好的Qwen2.5VL 32B Bot，可以体验效果。

使用Qwen2.5-14B-Instruct模型，大概3万条处理后的有效数据，loss降到了3.5左右的效果：

截图
🤖 部署到聊天机器人
AstrBot
AstrBot 是易上手的多平台 LLM 聊天机器人及开发框架 ✨ 平台支持 QQ、QQ频道、Telegram、微信、企微、飞书。

使用步骤：

部署 AstrBot
在 AstrBot 中部署消息平台
执行 weclone-cli server 启动api服务
在 AstrBot 中新增服务提供商，类型选择OpenAI，API Base URL 根据AstrBot部署方式填写（例如docker部署可能为http://172.17.0.1:8005/v1） ，模型填写gpt-3.5-turbo,API Key随意填写一个
微调后不支持工具调用，请先关掉默认的工具，消息平台发送指令： /tool off all，否则会没有微调后的效果。
根据微调时使用的default_system，在 AstrBot 中设置系统提示词。 5
Important

检查api_service的日志，尽量保证大模型服务请求的参数和微调时一致，tool插件能力都关掉。

调整采样参数，例如temperature、top_p、top_k等 配置自定义的模型参数
LangBot
LangBot 是一个开源的接入全球多种即时通信平台的 LLM 机器人平台，适合各种场景使用。

部署 LangBot
在 LangBot 中添加一个机器人
在模型页添加新模型，名称gpt-3.5-turbo，供应商选择 OpenAI，填写 请求 URL 为 WeClone 的地址，详细连接方式可以参考文档，API Key 任意填写。
image
在流水线配置中选择刚才添加的模型，或修改提示词配置
image
📌 路线图
 更丰富的上下文：包括上下文对话、聊天对象信息、时间等
 Memory 支持
 支持多模态:已支持图片
 数据增强
 支持GUI
 支持COT思考
问题解决
官方文档FAQ
同时建议使用DeepWiki解决问题。

❤️ 贡献代码
欢迎任何 Issues/Pull Requests！

你可以通过查看Issues或帮助审核 PR（拉取请求）来贡献。对于新功能的添加，请先通过 Issue 讨论。
开发环境：

uv pip install --group dev -e .
pre-commit install
项目使用pytest测试，pyright检查类型，ruff检查代码格式。

🙏 致谢
感谢以下代码贡献者和社区里其他成员的贡献


同时本项目受益于PyWxDump、LLaMA-Factory、AstrBot、LangBot等优秀开源项目。

⚠️ 免责声明
Caution

请勿用于非法用途，否则后果自负。

1. 使用目的
请用户慎重阅读并理解本免责声明的所有内容，确保在使用本项目时严格遵守相关规定。


⭐ Star History
Tip

如果本项目对您有帮助，或者您关注本项目的未来发展，请给项目 Star，谢谢

Star History Chart

克隆我们，保留灵魂的芬芳
About
🚀 One-stop solution for creating your digital avatar from chat history 💡 Fine-tune LLMs with your chat logs to capture your unique style, then bind to a chatbot to bring your digital self to life. 从聊天记录创造数字分身的一站式解决方案

weclone.love
Topics
llm qwen
Resources
 Readme
License
 AGPL-3.0 license
 Activity
Stars
 14.4k stars
Watchers
 72 watching
Forks
 1.1k forks
Report repository
Releases 9
v0.2.24
Latest
last week
+ 8 releases
Contributors
8
@xming521
@BAIKEMARK
@niulinbiao
@RockChinQ
@pre-commit-ci[bot]
@Mundi-Xu
@songhahaha66
@Copilot
Languages
Python
100.0%
Footer
© 2025 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact
Manage cookies
Do not share my personal information
