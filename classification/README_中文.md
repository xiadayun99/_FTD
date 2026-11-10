# FTD 论文复现包

解压后，将本目录中的文件和子目录放到 GitHub 仓库根目录。建议先在本地执行下面两条验证命令，生成文件默认写入 `build/`。

```bash
python -m pip install -r requirements-cpu.txt
python scripts/check_codec.py
python scripts/reproduce.py --output build
```

CPU 流程已从 50 阶段确认实验、54 阶段频率实验的 39,936 条正式请求和 3,780 条 Zstd 单元记录重建主文表 I–III、图 3 和补充表 S7。请求标识和启动周期采用公开别名。数值、候选分数与计时字段用于原样核算。

`python scripts/check_codec.py --cuda` 用小张量执行 CUDA 解码核验。完整 8B 重跑请按 [GPU_REPLAY.md](docs/GPU_REPLAY.md) 提供原始模型、三份适配器、四份 FP16 可变权重切片及其元数据，以及确认批次的冻结工作负载。脚本先逐文件核验哈希，再在新运行目录重建并准入纠正库。大权重和含原始题面文本的工作负载由作者另行提供。原始模型的文件哈希和必需路径见 `data/paper_assets.json`。

本包已验证 CPU 重算和编解码，GPU 计划与资产检查入口也已检查。CUDA 小测试及完整 8B 重跑需在目标设备执行。首次发布时由作者确定许可证，并按适用条款提供外部资产的取得方式。

