# CS 比赛预测与 Major 竞猜模拟

基于本地比赛记录、HLTV / Valve 积分快照，估计两队的逐图及 BO1 / BO3 / BO5 胜率；结合 BP 情景枚举，模拟 Major 瑞士轮并搜索竞猜填法。

完整的算法、参数、输入输出、验证与限制见 [项目说明](doc/PROJECT.md)。

## 安装与数据

在项目根目录执行，使用 Python 3.12 或以上版本：

```powershell
python -m pip install -r requirements.txt
```

运行依赖为 NumPy、SciPy。比赛数据放在 `data/team_maps/`，积分快照放在 `data/rankings/`。这些本地数据不随代码自动下载；仅安装依赖不能替代数据准备。

## 两队比赛预测

```powershell
python scripts/predict_bp.py FURIA Spirit --format all --date 2026-11-19
python scripts/predict_bp.py "Natus Vincere" "The MongolZ" --format BO3 --date 2026-11-19
```

输出默认写入 `reports/bp/队A__队B__日期/`，包括 `report.md`、`prediction.json` 和各赛制完整 BP 路径 CSV。

已知会打哪些地图时，可只查询条件胜率，不生成报告：

```powershell
python scripts/series_probability.py FURIA Falcons --maps Ancient,Dust2,Mirage --date 2026-11-19
```

积分引擎只有两种：默认 `--engine tier` 将 HLTV、Valve 各自分档；`--engine points` 将两榜都转换为 `log(1+积分)` 后求差。两队预测、Major 模拟和评估使用相同定义。

预测日期必须晚于比赛数据截止日。示例日期适用于当前截止到 2026-11-17 的本地数据；更新数据后请相应调整。

## Major 瑞士轮与竞猜

先检查种子表，每行一队，从高到低共 16 队：

```powershell
python scripts/major_sim.py --stage data/major/singapore2026/stage1.txt
```

默认模拟 120,000 次，搜索使“10 格至少答对 6 格”概率较高的填法。增加次数或同时比较五中线、六中线：

```powershell
python scripts/major_sim.py --stage data/major/singapore2026/stage1.txt --sims 400000 --threshold 5,6
```

Stage 3 全 BO3：

```powershell
python scripts/major_sim.py --stage data/major/singapore2026/stage3.txt --all-bo3
```

Stage 2/3 单阶段预测前，须把实际参赛队及种子顺序补齐为 16 行。需要提前模拟整届时：

```powershell
python scripts/major_sim.py --major data/major/singapore2026/stage1.txt data/major/singapore2026/stage2.txt data/major/singapore2026/stage3.txt --all-bo3
```

整届模式允许 Stage 2/3 只填 8 支直邀队，其余由上一阶段的模拟结果补齐。`--all-bo3` 在整届模式下只作用于 Stage 3；不加时，三个阶段都采用晋级/淘汰战 BO3、其余 BO1 的默认模式。

单阶段结果默认在 `reports/major/<种子表名>/`；整届结果默认在 `reports/major/singapore2026/`。

缺少模板时可运行 `python scripts/major_sim.py --init`。**该命令会覆盖三张种子表**，修改过名单后不要直接重复执行。

## 测试与评估

```powershell
python scripts/test/test_bp.py
python scripts/test/test_strength.py
python scripts/test/test_margin.py
python scripts/test/test_history_weights.py
python scripts/test/test_major.py
python scripts/evaluate.py
```

测试覆盖系列赛数学、BP 合法性、瑞士轮配对、BO3 切换及竞猜计分；评估脚本打印前向折样本外指标。参考项目对照方式见 [验证说明](doc/PROJECT.md#8-测试与评估)。

## 如何理解结果

- 通过率是固定填法在完整模拟中达到通过线的比例；每队战绩概率用于展示、生成搜索起点和对照，不用于独立相乘计算通过率。
- 搜索采用局部交换、随机重启与扰动，**不保证全局最优**；平均答对数最高也不等于通过率最高。
- 模拟结果取决于胜率模型、赛制和 BP 假设。当前积分快照回填历史存在信息泄漏，BP 行为权重也未经真实 veto 数据校准。
- 2026-11-19 已修正轮内 Buchholz 更新影响配对的问题。此前生成的 Major 报告应重新运行；增加模拟次数不能修复模型或规则错误。
- 地图表现使用双方积分难度加权净胜，该队每打任意一张图，所有旧图记录权重乘约 0.965936（20 张任意图减半），不做 `+8` 收缩。训练权重和 BP 窗口也不再依赖日历间隔；日期只用于历史顺序和防止未来信息泄漏。当前模型版本为 `all-map-count-20-v5`；旧缓存会自动避开，已有报告需重新生成。

对照此前逐图衰减与当前任意地图推进的衰减（需保留此前评估 JSON）：

```powershell
python scripts/evaluate.py --compare-history reports/evaluation/match_count_comparison.json --out reports/evaluation/all_map_count_comparison.json
```

现有对照报告是此前10张半衰期的历史测试，不代表当前20张版本的效果；20张版本已通过相关功能测试。详细结果见 [项目说明](doc/PROJECT.md#83-当前任意地图衰减模型对照)。
