# FTD ACL 切换历史深度补实验

这组对照复用投稿稿件中的 Qwen3-8B、GSM8K 和 DialogSum。每个任务使用原来冻结的 128 个测试输入，每个输入在各档配对比较。先重放常驻合并模型，然后在原 Incremental 切换轨迹的四个深度暂停切换，在固定权重状态下逐一解码相同的 128 个输入。每个请求新建 KV cache，使用原来的贪心解码和 256 token 上限。

| 已完成往返轮数 | 已完成切换次数 | 切回目标任务的来源 |
|---:|---:|---|
| 0 | 0 | 常驻合并参照 |
| 2 | 4 | QNLI |
| 8 | 16 | QNLI |
| 32 | 64 | QNLI |
| 128 | 256 | QNLI |

每一轮从目标任务切往中间适配器再切回。中间适配器按原轨迹交替使用 MRPC 和 QNLI。每个非零档都截取同一轨迹的前缀，且最后一次切回的来源固定。五档共 1,280 条生成记录，对应两个任务合计 256 个不同输入。两种任务分开汇总。

## 在原 RTX 服务器运行

在解压后的目录运行。

```bash
nohup bash run_switch_history.sh > ~/ftd_acl2027_switch_history_20261110.log 2>&1 &
tail -f ~/ftd_acl2027_switch_history_20261110.log
```

启动脚本默认使用以下原始环境和路径。

| 用途 | 默认路径 | 覆盖变量 |
|---|---|---|
| Python | `~/.conda/envs/llm/bin/python3` | `FTD_PYTHON` |
| GSM8K 原确认目录 | `~/ftd_acl2027_generation_20261109` | `FTD_GSM8K_RUN` |
| DialogSum 原确认目录 | `~/ftd_acl2027_dialogsum_20261109` | `FTD_DIALOGSUM_RUN` |
| 新结果目录 | `~/ftd_acl2027_switch_history_20261110` | `FTD_HISTORY_OUTPUT` |

每个原确认目录要保留完整的 `inputs/`、`completed/`、`stages/`、`trained/lora/` 和 `artifacts/references/`，以及 `run_config.json`、`runtime_contract.json` 和 `confirm_summary.json`。模型路径从原 `run_config.json` 读取。原结果压缩包只包含诊断记录，GPU 运行还需要服务器上的模型和权重。脚本只读取原目录，新的输出写在独立目录。

使用原环境的 RTX PRO 6000 Blackwell、PyTorch 2.10.0+cu128、CUDA 12.8、Triton 3.6.0、Transformers 4.57.6、PEFT 0.19.1。原运行时会核验完整 GPU 和软件算术配置，等待 GPU 连续空闲 30 秒，并检查至少 48 GiB 空闲显存及 104 GiB 可用主机内存。GPU 等待上限默认 8 小时，可用 `FTD_IDLE_TIMEOUT` 以秒覆盖。

## 结果

日志出现 `SEND_THIS_ARCHIVE` 后，上传以下文件。

```text
~/ftd_acl2027_switch_history_20261110/FTD_ACL_SwitchHistory_Results.tar.gz
```

`summary.csv` 每个任务、每档一行，报告逐 token 一致率、首次偏离位置、GSM8K 数值答案和正误变化、DialogSum ROUGE-L F1 变化，以及截断或非有限值情况。`requests.jsonl` 留存逐输入的结果。每档核验解码前后的完整可变权重哈希、该深度在原实验中已保存的哈希和对应请求，0 档另外逐一核对原常驻输出。结果包仅在两任务都通过核验后生成。

中断后运行同一条启动命令即可。已完成并通过文件校验的任务会复用，未完成任务会从常驻检查点重放，失败尝试保留在新目录下的 `_attempts/`。
