# History-4 experiment dashboard

> Snapshot: 2026-09-20 14:56 CST

This dashboard is generated from `experiments/history4/registry.json`.
`direct2` is the registry name for code-level `direct_token2`.

## Validity counts

| Level | Rows | Meaning |
|---|---:|---|
| A | 0 | 论文级：统一协议、有效 evaluator、三个训练 seed |
| B | 4 | 有效单 seed：统一协议、有效 evaluator，但只有一个训练 seed |
| C | 8 | 诊断级：结果文件有效，但存在架构、数据或训练口径混杂 |
| D | 5 | 无效：evaluator bug、缺结果文件，或实验未完成 |

## Results

| Benchmark | Task | Method | Checkpoint | Episodes | Score | Delta vs VPWEM | Level | Status |
|---|---|---|---:|---:|---:|---:|:---:|---|
| MIKASA | ShellGameTouch-v0 | nohistory | 600k | 100 | 88.00% | -4.00pp | B | complete |
| MIKASA | ShellGameTouch-v0 | direct | 600k | 100 | 95.00% | +3.00pp | B | complete |
| MIKASA | ShellGameTouch-v0 | direct2 | 600k | 100 | 94.00% | +2.00pp | B | complete |
| MIKASA | ShellGameTouch-v0 | vpwem | 600k | 100 | 29.00% | +0.00pp | D | invalid_evaluator |
| MIKASA | ShellGameTouch-v0 | vpwem | 600k | 100 | 92.00% | +0.00pp | C | diagnostic |
| MIKASA | ShellGameTouch-v0 | vpwem | 600k | 100 | 92.00% | +0.00pp | B | complete |
| MoMaRT | 10 configs multitask macro | vpwem | 400k | 300 | 63.67% | +0.00pp | C | complete_diagnostic |
| MoMaRT | 10 configs multitask macro | direct | 400k | 300 | 54.33% | -9.34pp | C | complete_diagnostic |
| MoMaRT | 10 configs multitask macro | direct2 | 400k | 300 | 68.00% | +4.33pp | C | complete_diagnostic |
| MoMaRT | 10 configs multitask macro | vpwem | 1M | 300 | 65.33% | +0.00pp | C | complete_diagnostic |
| MoMaRT | 10 configs multitask macro | direct2 | 1M | 300 | 71.67% | +6.34pp | C | complete_diagnostic |
| MoMaRT | 10 configs multitask macro | nohistory | 1M | 0 | — | — | D | evaluating |
| MoMaRT | 10 configs multitask macro | direct | 988k running (900k latest saved) | 0 | — | — | D | training_near_completion |
| Robomimic | Square/ph,mh + Transport/ph,mh macro | nohistory | 1M x 4 | 0 | — | — | D | trained_not_evaluated |
| Robomimic | Square/ph,mh + Transport/ph,mh macro | direct | Square 900k; Transport 400k | 0 | — | — | D | partial_not_evaluated |
| Robomimic | Square/ph,mh + Transport/ph,mh macro | direct | 400k | 120 | 32.50% | -46.70pp | C | complete_diagnostic |
| MIKASA | RememberColor3-v0 | direct | 20k/100k/400k evaluated; 600k trained | 90 | 10.00% | — | C | incomplete_curve |

## Execution plan

| Priority | ID | Action | Existing checkpoint | Acceptance gate |
|:---:|---|---|:---:|---|
| DONE | P0-01 | 已完成：BHR 重评 ShellGameTouch VPWEM 600k | yes | 100 paired episodes；reset online+EMA memory；结果文件含逐 episode success |
| P0 | P0-02 | 完成 MoMaRT NoHistory 1M 的 10x30 评测 | yes | 10/10 result.npy，seed=20260917；当前正在运行 |
| P0 | P0-03 | MoMaRT Direct 完成 1M 并跑 10x30 评测 | no | model_1000000.pt 稳定写完，10/10 result.npy |
| P0 | P0-04 | 评测 Robomimic NoHistory strict 1M 四任务 | yes | 每任务固定同一 100 paired episodes；统一 evaluator |
| P0 | P0-05 | 评测 Robomimic Direct strict 现有节点 | yes | Square 900k、Transport 400k，各先 30 episodes；宏平均 >=60% 才续训 |
| P0 | P0-06 | 补 MoMaRT NoHistory 400k 评测 | yes | 10x30、seed=20260917、与现有 400k 表同入口 |
| P1 | P1-01 | MoMaRT embedding 四方法严格单变量矩阵 | no | 同 dp_emb 数据、effective batch=256、1M、seed100；四方法 10x100 paired episodes |
| P1 | P1-02 | MIKASA ShellGameTouch 扩 seed200/300 | no | 仅在 P0-01 后执行；四方法 batch=64 重训，3 seeds x 100 paired episodes |
| P1 | P1-03 | Robomimic 统一四方法协议并新增 Direct2 | no | 同 history timestamps/sampler、同 8-layer、effective batch=64、1M；先 seed100 |
| P2 | P2-01 | 所有通过 seed100 gate 的组扩 seed200/300 | no | 报告 seed mean±std、Wilson CI、paired bootstrap/McNemar；禁止挑最佳 checkpoint |
| P2 | P2-02 | 统一效率审计 | yes | 参数量、history token、峰值显存、train step/s、推理延迟全部齐全 |

## Update workflow

1. Add or update one registry row only after checking its config, checkpoint and per-episode result.
2. Assign A/B/C/D using the declared validity definitions; a result file alone is not evidence of validity.
3. Run `python3 tools/history4_report.py` and commit the registry with its generated reports.
4. Run `python3 tools/history4_report.py --check` before using a table in a paper or slide.
