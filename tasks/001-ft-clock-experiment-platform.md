# Task 001 - Rebuild FT-Clock Experiment Platform

## Metadata

- Task ID: 001
- Title: Rebuild FT-Clock Experiment Platform
- Slug: ft-clock-experiment-platform
- Status: in_progress
- Type: refactor
- Priority: high
- Owner: agent
- Reviewer: pending
- Created At: 2026-03-10 23:10
- Updated At: 2026-03-11 21:32
- Branch: codex/refactor/t001-ft-clock-experiment-platform
- Worktree: ../wt-t001-ft-clock-experiment-platform
- Related Issues: N/A
- Related PRs: N/A
- Dependencies: none
- Follow-up Tasks: 002-core-ft-clock-runtime, 003-paper-results-pipeline, 004-mechanism-analysis-suite

---

## Goal

将当前以 `alpha/use_ft_eqm/use_nt_ft_fm` 为中心的实验代码，重构为面向论文的通用 FT-clock 实验平台。平台必须能用统一配置驱动 linear path、VP-style path、FT-clock、其他 schedule family、CIFAR-10/CIFAR-100、Heun/Euler、主图主表和机制分析，并输出论文就绪结果。

---

## Background / Context

- 当前连续模型训练在 `examples/image/training/train_loop.py` 中硬编码为线性 `CondOTProbPath`，FT-EqM 与 NT-FT-FM 只是特例分支，不具备通用 `path + clock` 结构。
- 当前评估与实验脚本在 `experiments/run_experiments.py`、`experiments/visualize_results.py` 中强绑定 `alpha/use_ft_eqm`，结果表结构过窄，只适合旧 alpha sweep。
- 当前数据集入口仅支持 `cifar10` 和 `imagenet`，不支持 `cifar100`。
- 最新研究笔记已经把目标语义明确为“严格的时间重参数化训练”，不是仅改推理 schedule。
- 当前主分支是 `main`，工作树干净；`tasks/` 下只有模板，没有正式任务文件。
- 当前 shell 环境未就绪：`python` 命令不存在，`python3` 环境里也缺少 `torchdiffeq`，执行阶段必须先补环境或绕开硬依赖。

---

## Scope

- 引入统一的 YAML-first 实验规格，覆盖 dataset、training、path、clock、solver、metrics、analysis、sweeps。
- 将连续训练目标重写为通用 FT-clock 语义：采样 `r`，计算 `s=ψ(r)`，构造 `z_r` 与 `\\tilde b_r`，回归重参数化条件速度。
- 支持两条 base path：`linear` 与 `trig_vp`。
- 支持 clock family：`uniform`、`ft_linear_beta`、`ft_vp_beta`、`poly_a0.5`、`poly_a2.0`、`cosine`、`sigmoid_k8`、`exp_l3`。
- 支持两类数据集：`cifar10`、`cifar100`。
- 支持主采样器 `heun2` 和对比采样器 `euler`。
- 将 NFE 统一定义为真实网络调用次数，并同时记录 `step_count`。
- 重写实验调度、结果记录、绘图和分析脚本，生成论文就绪 CSV/JSONL、PNG/PDF 和主表汇总。
- 覆盖实验 1-9 所需能力，但按父任务 + 子任务方式拆分执行。
- 增加最小测试集与 smoke checks，覆盖数学正确性、配置展开、NFE 计数和图表产物。

---

## Non-goals

- 不改 UNet/backbone 结构。
- 不扩展到 ImageNet 或 discrete flow 主线。
- 不保留旧 `--use_ft_eqm/--use_nt_ft_fm/alpha` 接口兼容性。
- 不在首轮实现里做全面多 seed 统计。
- 不把 SLURM/多机集群作为默认执行目标。

---

## Approach

- 平台主入口改为 YAML-first。CLI 只负责选择配置文件和做少量 override，不再让实验脚本拼接大段旧参数。
- 训练语义固定为严格重参数化训练：`r ~ U[0,1]`，`s=ψ(r)`，`z_r = α(s)x + σ(s)ε`，`\\tilde b_r = ψ'(r)[α'(s)x + σ'(s)ε]`。
- path 定义固定为闭式实现。`linear` 用 `α(s)=s, σ(s)=1-s`。`trig_vp` 用 `α(s)=sin(πs/2), σ(s)=cos(πs/2)`。
- clock 定义固定为闭式实现。`uniform: ψ(r)=r`。`ft_linear_beta: 1-(1-r)^{1/[2(1-β)]}`。`ft_vp_beta: (2/π) asin(1-(1-r)^{1/(1-β)})`。`poly_a0.5: 1-(1-r)^{0.5}`。`poly_a2.0: 1-(1-r)^2`。`cosine: 1-cos(πr/2)`。`sigmoid_k8` 为归一化 logistic。`exp_l3` 为归一化指数。
- 采样默认用 `heun2` 作为主结果 solver。solver 敏感性实验只比较 `euler` 与 `heun2`。所有横轴和主表中的 NFE 都按真实 model eval 统计。
- 结果记录采用 tidy long-form CSV/JSONL。每行表示一个 `run/checkpoint/eval-point/metric`，至少记录 `dataset`、`seed`、`path_family`、`clock_family`、`clock_param_name`、`clock_param_value`、`solver`、`nfe`、`step_count`、`metric`、`value`、`status`、`artifact_group`。
- 机制分析固定为离线分析脚本，不在训练期长期记录，不在常规 eval 中大规模落盘轨迹。
- 非 FT schedule family 固定代表参数，不额外做同级 sweep。公平性通过“统一 backbone、统一训练、统一 solver、统一 NFE 口径”保证。

---

## Execution Plan

1. 创建父任务 `001-ft-clock-experiment-platform`，作为总范围、总验收和总风险记录入口。
2. 创建子任务 `002-core-ft-clock-runtime`，负责通用 `path + clock + solver budget + dataset` 运行时重构。
3. 创建子任务 `003-paper-results-pipeline`，负责 YAML schema、runner、tidy results、主图主表、实验 1-5。
4. 创建子任务 `004-mechanism-analysis-suite`，负责实验 6-9 的离线分析、热力图、loss density、轨迹预算和 solver sensitivity。
5. 子任务 `002` 完成后，先做 `cifar10` 和 `cifar100` 最小 smoke 流程，再进入大 sweep。
6. 子任务 `003` 固定覆盖实验 1-5，子任务 `004` 固定覆盖实验 6-9。
7. Sweep 执行优先级固定为 `Exp1 -> Exp2/Exp4 -> Exp3/Exp5 -> Exp6-9`。如果算力或时间紧张，优先保证实验 1、2、4、6、9。
8. 每完成一个子任务或一个可回滚阶段，都提示可以提交 checkpoint commit。

---

## Test Plan

- 验证每个 clock 都满足 `ψ(0)=0`、`ψ(1)=1`、单调递增和导数符号正确。
- 验证 `linear` 与 `trig_vp` 的 `α,σ,α',σ'` 与 `\\tilde b_r` 公式，必要时用有限差分做数值比对。
- 验证 `euler/heun2` 的 NFE 预算换算后，记录到的真实 NFE 与 model wrapper 计数一致。
- 验证 YAML schema 能展开为确定性的 concrete runs，且 artifact 命名稳定。
- 验证 tidy result 行结构与聚合脚本一致，best-beta 选择逻辑、cross-path 汇总逻辑正确。
- 运行 `cifar10` 与 `cifar100` 的 one-batch smoke train/eval。
- 运行一条最小链路的 `train -> eval -> plot -> analyze` dry run。
- 用 synthetic fixture 结果文件回归测试主曲线、热力图和主表导出。

---

## Risks

- 当前环境缺少 `torchdiffeq`，且 `python` alias 不存在，执行阶段若不先修环境会直接阻塞。
- 破坏性重构会使旧 alpha-sweep 脚本和历史目录失去主线地位，需要明确迁移说明。
- `euler/heun2` 的 NFE 统一口径如果做错，会直接污染论文主结论。
- `Precision/Recall` 会增加依赖和计算开销，可能需要额外 metric backend 适配。
- `cifar100` 会显著增加训练成本，4 卡工作站下必须严格按优先级推进。
- 机制分析若直接存全轨迹会产生过大存储压力，因此必须坚持离线、抽样、按需分析。

---

## Open Questions

- None.

---

## Plan Self-Review

### Gaps

- 仓库当前没有测试体系，不能只靠手工跑通；子任务 `002` 必须顺手建立最小测试骨架。
- 父任务范围过大，不能直接作为单一实现单元执行。

### Feasibility

- 在 4 卡单机预算下可行，但必须按既定优先级分阶段推进。
- 现有代码已提供 affine path 与 ODE solver 基础，因此这次是结构重构，不是从零搭平台。

### Risks Review

- 最大工程风险是新旧语义并存后互相污染；本计划通过允许 breaking rewrite 避免双轨长期共存。
- 最大科研风险是把分析实验做得太早；本计划把机制分析独立成子任务并排在主结果之后。

### Should This Task Be Split?

- Yes.
- 本任务应作为父任务保留。
- 立即拆分为 `002-core-ft-clock-runtime`、`003-paper-results-pipeline`、`004-mechanism-analysis-suite` 三个子任务。

---

## Approval-Ready Summary

- 这是一次面向论文的实验平台重构，不是对旧 FT-EqM 代码做局部补丁。
- 目标平台以通用 `path + clock + solver + metric + analysis` 为核心，覆盖实验 1-9。
- 科学口径已经锁死：严格重参数化训练、uniform baseline、Heun 主结果、真实 NFE 计数、固定 schedule family 代表参数、离线机制分析、单 seed 首轮矩阵、4 卡单机、论文就绪产物。
- 为了可审查和可执行，必须按父任务 + 3 个子任务推进。

---

## Progress Log

- 2026-03-10 23:10 — Task planned with status: pending.
- 2026-03-10 23:18 — Execute stage started on `main`; parent task created directly as `in_progress`; child task creation and worktree bootstrap initiated.
- 2026-03-11 12:41 — Subtask slice 002 completed in parent branch: replaced the continuous training/evaluation path with generic path+clock runtime, added fixed-step Euler/Heun2 NFE accounting, added CIFAR-100 dataset support, and added initial runtime tests.
- 2026-03-11 13:55 — Subtask slice 003 completed in parent branch: rewrote the experiments layer around YAML configs, tidy CSV results, reusable aggregation helpers, paper plot/table generation, and FT-clock experiment config templates.
- 2026-03-11 13:55 — Subtask slice 004 completed in parent branch: added offline analysis utilities and analysis entrypoint for velocity/loss profiles and trajectory budget statistics, plus visualization regression tests.
- 2026-03-11 20:29 — Consistency remediation pass started after external review: correcting unconditional defaults, CFG real-NFE accounting, endpoint handling, legacy config drift, experiment matrix gaps, and heatmap/mechanism reproducibility details.
- 2026-03-11 20:35 — Consistency remediation pass completed: unconditional defaults now explicit, CFG counts real forward calls, analysis sampling avoids exact endpoints, mechanism configs include baselines, solver sensitivity covers linear and VP, legacy configs were migrated, and a heatmap-only plotting entrypoint was added.
- 2026-03-11 20:53 — Second remediation pass started: addressing auto best-beta resolution for downstream experiments, seed-aggregated statistics, eval sample-count correctness, analysis checkpoint reproducibility, README drift, and lightweight pipeline smoke tests.
- 2026-03-11 21:04 — Second remediation pass completed: E2/E3/E5 now resolve FT-best beta automatically from source experiment results, result aggregation now uses seed mean/std before best-beta selection, eval records actual sample counts, analysis supports checkpoint_epoch/checkpoint_path, README was migrated to the YAML-first workflow, and lightweight smoke tests passed without requiring runtime ML dependencies.
- 2026-03-11 21:25 — Third remediation pass completed: fixed double-aggregation in visualization/main-table export, added results.csv schema validation with fail-fast errors for legacy headers, added a conservative legacy-results migration script, and verified the fixes with regression tests plus a migration smoke run.
- 2026-03-11 21:32 — Fourth remediation pass completed: fixed the empty-results.csv edge case by forcing header creation for 0-byte files, added regression tests covering empty-file append/read flows, and re-ran the result-utils test suite.

---

## Decisions

- 使用 breaking rewrite，不保留旧 `alpha/use_ft_eqm/use_nt_ft_fm` 兼容层。
- 使用 YAML-first 作为主入口，CLI 仅作轻量封装。
- 主结果 solver 固定为 `heun2`，敏感性对比使用 `euler`。
- 真实 NFE 定义为网络实际前向调用次数。
- 父任务仅负责总控与集成，核心运行时、结果管线、机制分析分别下沉到子任务。
- `heun2` 在奇数 NFE 预算下采用 `heun2` 若干步加末尾 `euler` 一步的混合步法，以保持“真实网络调用次数”精确等于目标 NFE。
- continuous 主线不再通过 `torchdiffeq` 导入 solver；主实验使用本地 fixed-step 实现。
- 结果层统一采用 tidy CSV schema，并用 `experiments/result_utils.py` 作为 runner/plot/table 的共享聚合层。
- 机制分析固定通过离线 checkpoint 脚本完成，不在常规训练或 eval 中长期记录中间轨迹。
- legacy `alpha/use_ft_eqm` config keys 视为无效输入；runner 现在显式拒绝这类字段，避免旧配置被静默误用。
- CIFAR-10/100 的 FT-clock YAML 基线显式固定为无条件生成：`cfg_scale=0.0`、`class_drop_prob=1.0`。
- 机制分析除终端预算外，额外记录 `final_step_ratio` 与 `mean_curvature`，用于支撑 E8 的轨迹几何叙事。
- 下游配置通过 `best_beta_from` 自动解析 FT-best beta，避免 E2/E3/E5 手工同步 E1/E4 最优 beta。
- 主图主表的统计口径改为“先按 seed 聚合 mean/std，再选择 best beta”，禁止对多 seed 结果直接取最小 FID。
- 为了在无 `torch` 环境下保留最小回归保护，新增纯 Python smoke tests 覆盖 runner、checkpoint resolution、seed aggregation 和 eval batch replay 逻辑。
- `baseline_vs_best_beta` 新增 `already_aggregated` 语义，避免 visualize 层传入已聚合行后再次聚合导致 `std/num_seeds` 被压平。
- `results.csv` 现在强制校验表头是否等于当前 schema；遇到旧 `alpha_sweep` 结果文件会直接报错，而不是继续向错误表头追加新列。
- 提供 `experiments/migrate_legacy_results.py`，可把旧 `alpha`-schema 结果迁移到新 CSV 结构；迁移后的非 baseline 行统一标记为 `clock_family=legacy_alpha`，避免被误当作 FT-clock 主结果。
- `ensure_results_file` 现在同时覆盖“文件不存在”和“文件存在但 0 字节”两种初始化场景，避免第一条结果被写成伪表头。

---

## Working Notes

### Files Touched

- tasks/001-ft-clock-experiment-platform.md
- examples/image/train.py
- examples/image/train_arg_parser.py
- examples/image/models/model_configs.py
- examples/image/training/continuous_runtime.py
- examples/image/training/fixed_step_solver.py
- examples/image/training/metric_utils.py
- examples/image/training/train_loop.py
- examples/image/training/eval_loop.py
- tests/test_continuous_runtime.py
- tests/test_fixed_step_solver.py
- examples/image/training/analysis_utils.py
- experiments/result_utils.py
- experiments/checkpoint_utils.py
- experiments/run_experiments.py
- experiments/resolve_best_betas.py
- experiments/migrate_legacy_results.py
- experiments/visualize_results.py
- experiments/analyze_mechanisms.py
- experiments/configs/ft_clock/linear_main.yaml
- experiments/configs/ft_clock/cifar100_transfer.yaml
- experiments/configs/ft_clock/schedule_family.yaml
- experiments/configs/ft_clock/trig_vp.yaml
- experiments/configs/ft_clock/cross_path.yaml
- experiments/configs/ft_clock/mechanism_analysis.yaml
- experiments/configs/ft_clock/solver_sensitivity.yaml
- experiments/configs/main_results.yaml
- experiments/configs/alpha_sweep.yaml
- experiments/configs/alpha_sweep_ema.yaml
- tests/test_result_utils.py
- tests/test_visualize_results.py
- tests/test_checkpoint_utils.py
- tests/test_eval_utils.py
- tests/test_run_experiments.py
- environment.yml
- examples/image/requirements.txt
- README.md
- examples/image/README.md

### Notes

- 当前 examples/image 路径受 `torchdiffeq` 硬依赖影响，运行前必须解除入口级 import 阻塞或补装依赖。
- 当前仓库没有正式 task 文件，需要先补齐 lifecycle 所需任务记录层。
- `torchdiffeq` 对 examples/image 主线的入口级阻塞已解除，但 `flow_matching/solver/ode_solver.py` 仍保留原依赖，不再作为本任务主实验运行时。
- 本地系统 Python 缺少 `torch`、`torchvision`、`torchmetrics`、`yaml`、`sklearn`；因此本轮只能完成语法级检查，无法在当前 shell 直接跑训练或单测。
- 新增 `pyyaml` 依赖声明到 `environment.yml` 与 `examples/image/requirements.txt`，以支持 YAML-first runner。
- 由于当前系统 Python 仍缺训练依赖，runner/analysis/visualization 只完成了语法校验，尚未在真实 checkpoint 上执行。
- 外部审查发现旧 `experiments/configs/*.yaml` 仍残留 `alpha/use_ft_eqm` 命名；本轮已将主入口配置全部迁移到 `path_family/clock_family/clock_beta`，并给 legacy 文件加了迁移注释。
- `cifar100_transfer.yaml`、`schedule_family.yaml` 与 `cross_path.yaml` 已改为通过 `best_beta_from` 自动解析上游实验中的 FT-best beta，不再依赖手工替换。
- 2026-03-11 验证补充：`python3 -m py_compile ...` 通过，`python3 -m unittest tests/test_result_utils.py` 通过；`tests/test_continuous_runtime.py` 因缺少 `torch` 无法执行，`tests/test_visualize_results.py` 因缺少 `matplotlib` 无法执行。探针确认当前环境同时缺少 `torch`、`matplotlib`、`yaml`。
- 第二轮整改后，`best_beta_from` 已替代手工 beta 占位，`experiments/resolve_best_betas.py` 可将自动解析后的配置物化成独立 YAML 以便归档。
- 第二轮验证补充：`python3 -m unittest tests/test_result_utils.py tests/test_checkpoint_utils.py tests/test_eval_utils.py tests/test_run_experiments.py tests/test_visualize_results.py` 全部通过；这些测试通过 stub 避开了本机缺失的 `torch` 和 `matplotlib`，但仍不能替代真实训练/eval smoke run。
- 第三轮验证补充：`python3 -m unittest tests/test_result_utils.py tests/test_visualize_results.py` 通过，覆盖了“已聚合输入不再二次聚合”和“legacy schema fail-fast”；`python3 experiments/migrate_legacy_results.py --src experiments/results/alpha_sweep/results.csv --out <tmp>` smoke run 成功。
- 第四轮验证补充：`python3 -m unittest tests/test_result_utils.py` 通过，新增覆盖“空文件补表头”和“空文件后 append 仍可读”。
