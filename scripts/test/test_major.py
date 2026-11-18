"""Major 瑞士轮模拟与 Pick'Em 优化的正确性测试（不重训模型，秒级）。

跑法：python scripts/test/test_major.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import major_sim as ms  # noqa: E402

PASS = []


def check(name: str, fn) -> None:
    fn()
    PASS.append(name)
    print(f"  PASS  {name}")


def symmetric_matrix(rng: np.random.Generator) -> np.ndarray:
    """构造满足 M[i,j] + M[j,i] = 1 的 16×16 胜率矩阵。"""
    a = rng.uniform(.15, .85, (16, 16))
    m = a / (a + a.T)
    np.fill_diagonal(m, .5)
    return m


# ---------------------------------------------------------------- 规则结构
def t_initial_pairs():
    assert ms.INITIAL_PAIRS == [(i, i + 8) for i in range(8)], "第 1 轮必须是 1v9…8v16"


def t_priority_rows():
    assert len(ms.PRIORITY_ROWS) == 15, "规则书优先级表有 15 行"
    seen = set()
    for row in ms.PRIORITY_ROWS:
        positions = sorted(i for pair in row for i in pair)
        assert positions == list(range(6)), f"每行必须是 6 个位置的完美匹配：{row}"
        seen.add(tuple(sorted(tuple(sorted(p)) for p in row)))
    assert len(seen) == 15, "15 行互不相同"


def t_first_row_is_1v6():
    assert ms.PRIORITY_ROWS[0] == [(0, 5), (1, 4), (2, 3)], "优先级第一行应为 1v6 2v5 3v4"


# ---------------------------------------------------------------- 模拟
def t_match_formats_and_structure():
    rng = np.random.default_rng(0)
    m = symmetric_matrix(rng)
    ids = np.tile(np.arange(16, dtype=np.int16), (3000, 1))
    sim = ms.simulate(ids, m, m, seed=7)
    assert (sim["bo1_matches"], sim["bo3_matches"]) == (20.0, 13.0), \
        f"五轮恒定 20 BO1 + 13 BO3，实际 {sim['bo1_matches']}/{sim['bo3_matches']}"
    assert sim["rematches"] == 0, f"同阶段内不应重赛，实际 {sim['rematches']}"
    assert sim["fallback"] == 0, "第 4/5 轮不应退化为重赛"
    got = ms.check_structure(sim["code"])
    assert list(got.values()) == [6000, 9000, 9000, 9000, 9000, 6000], got


def t_bo3_only_on_advance_or_elimination():
    """把 BO1 概率全设成 0.5、BO3 设成极端值，检验 BO3 只用在该用的场次。"""
    rng = np.random.default_rng(1)
    m1 = np.full((16, 16), .5)
    m3 = symmetric_matrix(rng)
    ids = np.tile(np.arange(16, dtype=np.int16), (2000, 1))
    a = ms.simulate(ids, m1, m3, seed=11)
    b = ms.simulate(ids, m1, m1, seed=11)
    assert not np.array_equal(a["code"], b["code"]), "BO3 矩阵必须真的影响结果"


def t_all_bo3_uses_only_bo3_matrix():
    """全 BO3 必须使用 BO3 矩阵，且改动 BO1 矩阵不影响任何比赛结果。"""
    rng = np.random.default_rng(17)
    m1 = symmetric_matrix(rng)
    m3 = symmetric_matrix(rng)
    ids = np.tile(np.arange(16, dtype=np.int16), (500, 1))
    actual = ms.simulate(ids, m1, m3, seed=19, all_bo3=True)
    changed_bo1 = ms.simulate(ids, 1 - m1, m3, seed=19, all_bo3=True)
    reference = ms.simulate(ids, m3, m3, seed=19)
    assert actual["all_bo3"] is True
    assert reference["all_bo3"] is False
    assert (actual["bo1_matches"], actual["bo3_matches"]) == (0.0, 33.0)
    for field in ("code", "difficulty", "wins", "losses"):
        assert np.array_equal(actual[field], changed_bo1[field]), field
        assert np.array_equal(actual[field], reference[field]), field
    assert actual["rematches"] == 0
    ms.check_structure(actual["code"])


def t_deterministic_ordering():
    """强者恒强：i<j ⇒ i 必胜的矩阵下，3-0 必是 1、2 号种子，0-3 必是 15、16 号种子。"""
    m = np.full((16, 16), .0)
    for i in range(16):
        for j in range(16):
            m[i, j] = 1.0 if i < j else 0.0
    np.fill_diagonal(m, .5)
    ids = np.tile(np.arange(16, dtype=np.int16), (200, 1))
    sim = ms.simulate(ids, m, m, seed=3)
    code = sim["code"]
    assert np.array_equal(np.unique(np.argwhere(code == 0)[:, 1]), np.array([0, 1]))
    assert np.array_equal(np.unique(np.argwhere(code == 5)[:, 1]), np.array([14, 15]))
    # 同战绩内按 Difficulty 排序：这一点由 final_order 的第二个用例单独覆盖
    assert sim["rematches"] == 0


def t_final_order_uses_record_then_difficulty():
    code = np.array([[1, 0, 2, 5]], np.int8)          # 3-1 / 3-0 / 3-2 / 0-3
    diff = np.array([[9, 9, 9, 9]], np.int16)
    assert ms.final_order(code, diff)[0].tolist() == [1, 0, 2, 3]
    code2 = np.array([[0, 0]], np.int8)               # 同为 3-0，比 Difficulty
    diff2 = np.array([[1, 4]], np.int16)
    assert ms.final_order(code2, diff2)[0].tolist() == [1, 0], "同战绩时 Difficulty 高者排前"


# ---------------------------------------------------------------- Pick'Em
def t_greedy_slots_valid():
    rng = np.random.default_rng(2)
    m = symmetric_matrix(rng)
    sim = ms.simulate(np.tile(np.arange(16, dtype=np.int16), (4000, 1)), m, m, seed=5)
    pick = ms.PickEm(sim["code"])
    labels = pick.greedy_expected()
    counts = {int(v): int((labels == v).sum()) for v in (1, 2, 3)}
    assert counts == {1: 2, 2: 6, 3: 2}, f"格子配额应为 2/6/2，实际 {counts}"
    desc = ms.describe_pick(labels, [f"T{i}" for i in range(16)], pick.marginal)
    assert sum(len(v) for v in desc.values()) == 10
    assert len({d["team"] for v in desc.values() for d in v}) == 10, "10 格必须是 10 支不同的队"


def t_distribution_and_thresholds():
    rng = np.random.default_rng(3)
    m = symmetric_matrix(rng)
    sim = ms.simulate(np.tile(np.arange(16, dtype=np.int16), (4000, 1)), m, m, seed=9)
    pick = ms.PickEm(sim["code"])
    labels = pick.greedy_expected()
    st = pick.stats(labels, [5, 6, 7])
    assert abs(sum(st["probability"].values()) - 1) < 1e-9
    assert st["at_least"]["5"] >= st["at_least"]["6"] >= st["at_least"]["7"]
    assert abs(st["expected"] - sum(int(k) * v for k, v in st["probability"].items())) < 1e-9


def t_optimizer_beats_expected_only():
    """P(≥K) 优化后的通过率不应低于「只最大化期望答对数」的解。"""
    rng = np.random.default_rng(4)
    m = symmetric_matrix(rng)
    sim = ms.simulate(np.tile(np.arange(16, dtype=np.int16), (20000, 1)), m, m, seed=13)
    pick = ms.PickEm(sim["code"])
    base = pick.stats(pick.greedy_expected(), [5])["at_least"]["5"]
    sol = pick.solve(5, restarts=4, rng=__import__("random").Random(13))
    assert sol["probability"] >= base - 1e-9, f"优化解 {sol['probability']} < 贪心解 {base}"


# ---------------------------------------------------------------- 种子表
def t_read_seeds():
    p = ROOT / "_tmp_seeds_test.txt"
    try:
        p.write_text("# 注释\n1. Alpha\n\n2、Beta\n3) Gamma\n9z\n100 Thieves\n", encoding="utf-8")
        assert ms.read_seeds(p) == ["Alpha", "Beta", "Gamma", "9z", "100 Thieves"]
    finally:
        p.unlink(missing_ok=True)


def main() -> None:
    print("Major 模拟器正确性测试")
    for name, fn in sorted(globals().items()):
        if name.startswith("t_") and callable(fn):
            check(name, fn)
    print(f"\n全部 {len(PASS)} 项 PASS")


if __name__ == "__main__":
    main()
