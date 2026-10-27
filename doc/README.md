# doc/ 索引

**命名约定。** `YYYY-MM-DD_<试验主要内容>.md` —— 日期在前，`ls` 出来天然就是时间线。
日期取该试验的**起始日**；跨天的 run log 用开始日，不在文件名里表示区间（区间写进文档正文）。
阶段号（P0/P1/P1b/P2/P3/P4/P6…）保留在内容段里，因为它标识的是**工程阶段**而不是时间。

`login.md` 是凭据文件，**不是试验记录**，不参与本约定；它含明文 SSH 密码，被 `.gitignore`
排除，永不入库。本文件（索引）同样不是试验记录，故用惯例名 `README.md`。

## 已交付

| 文件 | 日期 | 阶段 | 一句话 |
|---|---|---|---|
| [2026-09-16_P0_interface_audit.md](2026-09-16_P0_interface_audit.md) | 09-16 | P0 | TD 臂与 VQ-BeT 臂**只差一个动作头**的证据：文件级恒等（仅包名）+ harness duck-typing 契约 |
| [2026-09-16_P1_action_anchor_audit.md](2026-09-16_P1_action_anchor_audit.md) | 09-16 | P1 | PushT 动作语料、`LinearNormalizer(limits)`、k-means 锚点，以及决定上限的 **anchor floor** |
| [2026-09-16_P1b_diffusion_scheduler_audit.md](2026-09-16_P1b_diffusion_scheduler_audit.md) | 09-16 | P1b | 自实现截断 DDIM 与 `diffusers` 逐字段对齐；把「2 步 ≈ 1 步」从猜测变成可验证结论 |
| [2026-09-16_P2_small_set_overfit.md](2026-09-16_P2_small_set_overfit.md) | 09-16 | P2 | 32/128 样本过拟合：先证明头**能**学，再谈长训练 |
| [2026-09-16_P3_smoke_test.md](2026-09-16_P3_smoke_test.md) | 09-16 | P3 | 走通真实的 train→ckpt→rollout 路径 |
| [2026-09-16_P4_run_log.md](2026-09-16_P4_run_log.md) | 09-16 | P4 | K=20 单 seed 快速验证的运行记录（含 n=100 补测附录 A/B、续训到 ep100 的附录 C） |
| [2026-09-17_K_sweep_run_log.md](2026-09-17_K_sweep_run_log.md) | 09-17 | K sweep | K∈{5,10,40} 三次并发训练 + 12 次 n=100 rollout 的纯记录（§12 续训可行性、§13 续训执行） |
| [2026-09-18_DP_orig_run_log.md](2026-09-18_DP_orig_run_log.md) | 09-18 | DP 臂 | **上游自己的** Diffusion Policy 原样跑第三臂的运行记录 |
| [2026-09-19_ablation_record.md](2026-09-19_ablation_record.md) | 09-19 | 总记录 | 两批消融的数字集中到一处：**(A)** K 消融 4 组、**(B)** 三臂动作头 |
| [2026-09-20_400ep_run_log.md](2026-09-20_400ep_run_log.md) | 09-20 | 400ep | K=20/K=40 各 400 epoch **从头**训练 + 每 5 ep 一次 n=100 rollout 的运行记录；**09-21 16:22 终止于 ep137/ep92**（§11）。含「计时无效」的原因（§3）与**训练速度实测**（§10） |
| [2026-10-27_attn_sdpa.md](2026-10-27_attn_sdpa.md) | 10-27 | Kernel | attention 手写链 ↔ SDPA 的双实现：parity（fp32 舍入级、bf16 4.9e-4）、mask 语义、**实际选中的 backend**（本机 mem-efficient，非 flash），以及**为什么 FM 线的 2.29× 不能外推到本仓库** |

## 尚未交付（计划中的名字按同一约定）

| 阶段 | 计划文件名 | 内容 |
|---|---|---|
| P5 | `2026-09-XX_P5_infer_steps_benchmark.md` | 1 步 vs 2 步推理（同 ckpt，n=100） |
| P6 | `2026-09-XX_P6_vs_vqbet.md` | TD / VQ-BeT / DP 三臂 100 epoch 正式对比 |
| D0 | `2026-09-XX_D0_dp_interface_audit.md` | DP 臂接入 harness 的静态审计 |
| D2/D3 | `2026-09-XX_D2D3_dp_orig_run_log.md` | DP 臂 D2/D3 运行记录 |

> ⚠️ `configs/*.yaml` 里指向 `patch_policy_td_vs_vqbet.md` 的注释是**旧名**，对应的 P6 文档
> 尚未交付，所以没有跟着改名 —— 交付时按上表的新名写，并顺手改掉那 5 处 config 注释。

## 读的顺序

想搞清楚**这个头是怎么被验证到可训的**：P0 → P1 → P1b → P2 → P3。
想只看**结果**：[2026-09-19_ablation_record.md](2026-09-19_ablation_record.md)，它给出了每
个数字的出处与用的哪一套评测。想复现某次运行：[2026-09-16_P4_run_log.md](2026-09-16_P4_run_log.md)
与 [2026-09-17_K_sweep_run_log.md](2026-09-17_K_sweep_run_log.md) 是纯记录，正文带 read-back 命令。
想知道**正在跑的那一跑**（400 epoch、K=20/K=40、每 5 ep 一次 n=100）：
[2026-09-20_400ep_run_log.md](2026-09-20_400ep_run_log.md) —— 它同时是这个项目里
**训练速度**（§10，K=20 830.0 s/epoch、K=40 1237.0 s/epoch、VQ-BeT 参照线 878.8 s/epoch）
与这一跑**「rollout 计时为什么无效」**（§3）两件事的唯一出处。
引用本跑的毫秒之前先读 §3；引用训练墙钟之前先读 §10（两者结论相反，别串了）。
