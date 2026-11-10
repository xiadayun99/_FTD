# FTD ACL 2027 复现包

解压后，将仓库根目录的文件和子目录上传到 GitHub。根目录应直接看到
README.md、generation、evidence、classification、scripts 和 docs。
请一并保留 .gitattributes 与 .gitignore。

## CPU 核验

在仓库根目录使用 Python 3.10 或更高版本。

~~~bash
python scripts/reproduce.py --output build/generation
~~~

这条命令从随包记录重算 GSM8K 和 DialogSum 的正文表 1、2、
附录表 3、4、附录 D 的固定回切来源五档对照表，以及文中使用的数值宏，并和终稿逐文件比对。
生成图 3 和核验附录 C 时，建议使用 Python 3.12。

~~~bash
python -m pip install -r requirements-cpu.txt
python scripts/reproduce.py --output build/generation --with-figure
python classification/scripts/check_codec.py
python classification/scripts/reproduce.py --output build/classification
~~~

## GPU 复跑

按 [GPU 复跑指南](docs/GPU_REPLAY.md) 准备原 Qwen3-8B、本次论文使用的
MRPC/QNLI LoRA 以及 base、MRPC、QNLI 的原始 FP16 参考切片。
先运行 check-assets 核对输入，再依次运行 pilot 和 confirm。
完整复跑需要这些外部权重。CPU 核验直接使用本包记录。
最后一轮补实验的 GPU 入口见 [切换历史深度复跑说明](README_switch_history_zh.md)。

若将软件提供给 ARR 匿名评审，请使用匿名仓库或按投稿表单分交
匿名软件与数据。原始运行目录的日志和参考元数据可能包含主机标识。
