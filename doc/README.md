# 当前项目入口

计算**任意两队的 BO1 / BO3 / BO5 胜率**，地图感知：先算七图胜率，再用 BP 规则枚举出图集合，
按集合积成系列赛概率。

- 算法、公式、参数、验证数字：[ALGORITHM.md](ALGORITHM.md)
- 设计取舍与历史：[DESIGN.md](DESIGN.md)
- 已停用的旧方案：`archive/v5/`

## 快速开始

```powershell
python -m pip install -r requirements.txt          # 常预测的话只需要 numpy + scipy
python scripts/predict_bp.py FURIA Spirit --format all
python scripts/predict_bp.py "Natus Vincere" "The MongolZ" --format BO3
```

输出写到 `reports/bp/队A__队B__日期/`（`prediction.json`、`report.md`、各赛制全部路径 CSV）。

不落盘的单次查询与条件预测：

```powershell
python scripts/series_probability.py FURIA Spirit
python scripts/series_probability.py FURIA Falcons --maps Ancient,Dust2,Mirage
```

## 验证

```powershell
python scripts/evaluate.py     # 前向折 OOF：地图级 + 系列赛级，tier / points 对照
python scripts/test_bp.py      # 枚举合法性、归一、互补、解析真值
```

## 常用参数

| 参数 | 说明 |
|---|---|
| `--format both\|all\|BO1\|BO3\|BO5` | `both` = BO1+BO3（默认）；`all` = 三种 |
| `--engine tier\|points` | 默认 `tier`（分档差）；`points` 用原始积分差 |
| `--tier-width` / `--tier-slope` | 档宽基数与随积分的增长率（默认 80 / 0.06）|
| `--out 目录` | 改写输出根目录，默认 `reports/bp` |
| `--top N` | 报告里列多少条 Top 路径 |

队名含空格时加英文双引号；在 `scripts/` 目录内运行时去掉命令里的 `scripts/` 前缀。
预测日期必须晚于数据截止日（当前 2026-11-17）。
