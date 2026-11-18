"""Major 瑞士轮蒙特卡洛模拟 + 游戏内 Pick'Em 最优作业。

规则来源（Valve《CS Major Supplemental Rulebook》，32 队三阶段版本，2025-01 起）：
  * 每阶段 16 队瑞士轮、五轮；三胜晋级、三负淘汰。
  * 同战绩对同战绩；**晋级战与淘汰战为 BO3，其余全部 BO1**。
  * 第 1 轮固定 1v9 2v10 3v11 4v12 5v13 6v14 7v15 8v16。
  * 第 2、3 轮：组内按种子排序，"最高种子 vs 最低种子中尚未交手者"。
  * 第 4、5 轮：组内恰好 6 队，按规则书 15 行优先级表（1v6 2v5 3v4 …）取第一条
    不产生重赛的行；全部行都会重赛时退化为"重赛最少"的行。
  * 阶段内种子 =（当前战绩，Difficulty/Buchholz，本阶段初始种子）。
    Difficulty = 已交手对手的（胜 − 负）之和：配对时用本轮开打前的战绩，
    阶段结束时用对手的最终战绩（用于阶段 2/3 的 9-16 号种子序与季后赛 1-8 号种子序）。
  * 阶段 2/3：直邀 8 队为种子 1-8（按赛前全球 VRS 序）；上一阶段晋级 8 队为种子 9-16
    （按上一阶段的最终种子序）。

游戏内 Pick'Em（Cologne 2026 Major Hub 起，新加坡 2026 沿用）：
  * 每阶段 10 格：2 个 3-0、6 个 3-1 或 3-2 晋级、2 个 0-3。
  * 答对 **6 个（十中六）** 才算通过本阶段，纪念币升一级（青铜→白银→黄金→钻石，每级 300 tokens）。
    六格晋级若答对 5 个也不能单独救回一个错格——按总答对数算。
  * 槽位不能串用：把 3-0 队填进晋级格，该格不得分；反之亦然。
  * 16 队五轮瑞士轮的结果分布恒定：2 个 3-0、3 个 3-1、3 个 3-2、3 个 2-3、
    3 个 1-3、2 个 0-3。所以 3-0 与 0-3 各只有 2 个正确答案，10 格的理论上限因此很低。

本脚本：
  可选 --all-bo3：单阶段时当前阶段全 BO3；整届时仅 Stage 3 全 BO3。
  1) 复用项目现有的逐图胜率 + 规则 veto 引擎算两两 BO1/BO3 胜率，按上述规则跑蒙特卡洛，
     给出每队 3-0 / 3-1 / 3-2 / 0-3 概率与晋级概率；
  2) 在模拟出的**联合分布**上直接求解"使 P(答对 ≥ K 个) 最大"的 10 个格子，
     并给出每格边际命中率、与"最大化期望答对数"和"按种子硬填"的对照。

用法见 --help 或 doc/MAJOR.md。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_io import key, load_maps, load_rankings  # noqa: E402
from features import build_training_features, predict_features  # noqa: E402
from predict_bp import enumerate_paths, summarize_paths  # noqa: E402
from series_probability import (RAIL, TIER_SLOPE, TIER_WIDTH, build_profiles,  # noqa: E402
                                build_tiers, fit_strength, strength_row, tier_second)

CONFIG = ROOT / "configs/bp_rules.json"
TEMPLATE_DIR = ROOT / "data/major/singapore2026"

# ---------------------------------------------------------------- 瑞士轮结构
N_TEAMS = 16
INITIAL_PAIRS = [(0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)]

# 规则书第 4/5 轮优先级表（组内 6 队已按种子排好，位置 0-5）
PRIORITY_ROWS = [
    [(0, 5), (1, 4), (2, 3)], [(0, 5), (1, 3), (2, 4)], [(0, 4), (1, 5), (2, 3)],
    [(0, 4), (1, 3), (2, 5)], [(0, 3), (1, 5), (2, 4)], [(0, 3), (1, 4), (2, 5)],
    [(0, 5), (1, 2), (3, 4)], [(0, 4), (1, 2), (3, 5)], [(0, 2), (1, 5), (3, 4)],
    [(0, 2), (1, 4), (3, 5)], [(0, 3), (1, 2), (4, 5)], [(0, 2), (1, 3), (4, 5)],
    [(0, 1), (2, 5), (3, 4)], [(0, 1), (2, 4), (3, 5)], [(0, 1), (2, 3), (4, 5)],
]

# 结果编码：序号即强弱排名，阶段内最终种子直接用序号排序
OUTCOMES = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
CODE_OF = {(3, 0): 0, (3, 1): 1, (3, 2): 2, (2, 3): 3, (1, 3): 4, (0, 3): 5}
STRUCTURAL_COUNTS = np.array([2, 3, 3, 3, 3, 2], np.int64)

# 新加坡 2026 实际参赛名单（HLTV 2026-11-02 邀请名单 + Valve 全球 VRS 2026_11_02）
SINGAPORE_2026 = {
    "stage1": ["Ninjas in Pyjamas", "PARIVISION", "HEROIC", "FaZe", "Liquid", "Eternal Fire",
               "100 Thieves", "9z", "M80", "DENDELE", "Imperial", "Wildcard",
               "BC.Game", "FlyQuest", "Chinggis Warriors", "Lynn Vision"],
    "stage2_invited": ["FUT", "Nemiga", "G2", "1win", "Natus Vincere", "Astralis",
                       "The MongolZ", "MIBR"],
    "stage3_invited": ["MOUZ", "Vitality", "FURIA", "Aurora", "Spirit", "BETBOOM",
                       "Falcons", "Legacy"],
}
EVENT_NOTE = ("PGL Major Singapore 2026（2026-11-24 ~ 12-13，32 队三阶段瑞士轮 + 8 队单败季后赛；"
              "种子 = Valve 全球 VRS 2026-11-02）")


# ================================================================ 种子表
def read_seeds(path: Path) -> list[str]:
    """每行一队，从高到低；忽略空行与 # 注释，容忍 "1. " / "1、" / "1) " 前缀。"""
    teams: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^\d+\s*[.、):：]\s*", "", line).strip()
        if line:
            teams.append(line)
    return teams


def write_templates(out_dir: Path = TEMPLATE_DIR) -> list[Path]:
    """写出新加坡 2026 三张种子表（阶段 2/3 只有直邀的 1-8 号）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    head = ("# {title}\n"
            "# 每行一支队伍，按官方种子从高到低（第 1 行 = 1 号种子）。\n"
            "# '#' 开头为注释，空行忽略；也可写 '1. MOUZ' 这种带编号的形式。\n"
            "# 改完直接跑：python scripts/major_sim.py --stage {rel}\n"
            "# 想看更稳/更准的解，加 --sims 400000\n")
    out: list[Path] = []

    s1 = out_dir / "stage1.txt"
    s1.write_text(
        head.format(title=f"{EVENT_NOTE} · Stage 1 种子表（16 队全填）",
                    rel="data/major/singapore2026/stage1.txt")
        + "# 16 队齐 = 直接可跑；替换成你自己的种子序即可。\n\n"
        + "\n".join(SINGAPORE_2026["stage1"]) + "\n", encoding="utf-8")
    out.append(s1)

    for name, label, invited, prev in [
            ("stage2", "Stage 2", SINGAPORE_2026["stage2_invited"], "Stage 1"),
            ("stage3", "Stage 3", SINGAPORE_2026["stage3_invited"], "Stage 2")]:
        p = out_dir / f"{name}.txt"
        p.write_text(
            head.format(title=f"{EVENT_NOTE} · {label} 种子表",
                        rel=f"data/major/singapore2026/{name}.txt")
            + f"# 种子 1-8 = 直邀队（按赛前全球 VRS 序），已填好。\n"
            f"# 种子 9-16 = {prev} 的 8 支晋级队，按 {prev} 最终种子序；\n"
            f"#   留空（只写 8 行）时脚本不会自动猜——请在该阶段开打前把 16 行填满。\n\n"
            + "\n".join(invited) + "\n", encoding="utf-8")
        out.append(p)
    return out


# ================================================================ 两两胜率
class Pairwise:
    """一次装载模型，反复算两两 BO1/BO3/BO5 胜率（带磁盘缓存）。"""

    def __init__(self, as_of: str, engine: str = "tier", rail: float = RAIL,
                 tier_width: float = TIER_WIDTH, tier_slope: float = TIER_SLOPE,
                 cache_dir: Path | None = None, verbose: bool = True):
        self.as_of, self.engine, self.rail, self.verbose = as_of, engine, rail, verbose
        t0 = time.time()
        self.obs, self.names, manifest, _, _ = load_maps()
        self.rankings, self.medians, rank_manifest = load_rankings()
        self.rows, self.model = build_training_features(self.obs, self.rankings, self.medians)
        self.model.update(names=self.names, trained_through=self.rows[-1]["date"])
        if as_of <= self.model["trained_through"]:
            raise ValueError(f"日期须晚于训练截止日 {self.model['trained_through']}")
        self.tiers, self.fallback = build_tiers(self.rankings, self.medians, tier_width, tier_slope)
        self.second = tier_second(self.tiers, self.fallback) if engine == "tier" else None
        self.strength = fit_strength(self.rows, self.second)
        self.w, self.c = self.strength["weights"], self.strength["c"]
        self.cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.pool = self.cfg["maps"]
        self.fingerprint = hashlib.sha256(json.dumps(dict(
            as_of=as_of, engine=engine, rail=rail, width=tier_width, slope=tier_slope,
            data=[m["sha256"] for m in manifest + rank_manifest],
            bp=hashlib.sha256(CONFIG.read_bytes()).hexdigest()), sort_keys=True).encode()).hexdigest()[:16]
        self.cache_dir = cache_dir or (ROOT / "reports/major/_pairwise")
        self.cache_path = self.cache_dir / f"pairwise_{self.fingerprint}.json"
        self._probs: dict[tuple[str, str], dict[str, float]] = {}
        self._load_cache()
        if verbose:
            print(f"[模型] {time.time() - t0:.1f}s 训练截止={self.model['trained_through']} "
                  f"引擎={engine} 护栏={rail} 指纹={self.fingerprint}", flush=True)

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") == self.fingerprint:
                self._probs = {tuple(k.split("|")): v for k, v in payload["probs"].items()}
                if self.verbose:
                    print(f"[缓存] {len(self._probs)} 组对阵", flush=True)
        except (ValueError, KeyError):
            self._probs = {}

    def save_cache(self) -> None:
        if not self._probs:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(dict(
            fingerprint=self.fingerprint, as_of=self.as_of, engine=self.engine, rail=self.rail,
            probs={f"{a}|{b}": v for (a, b), v in sorted(self._probs.items())}), ensure_ascii=False),
            encoding="utf-8")

    # ---------------------------------------------------- 已知性
    def resolve(self, team: str) -> str:
        k = key(team)
        if k not in self.names:
            raise ValueError(f"数据里没有这支队伍：{team!r}（检查拼写；用 --init 可看标准写法）")
        return k

    # ---------------------------------------------------- 单对
    def map_probs(self, a: str, b: str) -> dict[str, float]:
        a, b = key(a), key(b)          # 允许传显示名
        col2 = self.second(dict(a=a, b=b)) if self.second else None
        out = {}
        for m in self.pool:
            x, av, _, _ = predict_features(self.model, a, b, m, self.as_of)
            z = float(strength_row(x, col2 if self.second else x[1]) @ np.array(self.w) - self.c * av)
            raw = expit(z)
            out[m] = float(min(max(raw, self.rail), 1 - self.rail)) if self.rail else float(raw)
        return out

    def probs(self, a: str, b: str, formats=("BO1", "BO3")) -> dict[str, float]:
        """p(a 赢 b)。方向严格互补：p(b,a) = 1 − p(a,b)。允许传显示名。"""
        a, b = key(a), key(b)
        pair = (a, b)
        res = dict(self._probs.get(pair, {}))
        missing = [f for f in formats if f not in res]
        if missing:
            mp = self.map_probs(a, b)
            profiles = build_profiles(a, b, self.as_of, self.model, self.rows, self.pool)
            for fmt in missing:
                summary = summarize_paths(enumerate_paths(fmt, profiles, mp, self.cfg), mp, 1)
                res[fmt] = float(summary["p_a"])
            self._probs[pair] = res
        return {f: res[f] for f in formats}

    def matrix(self, teams: list[str], fmt: str, progress: bool = True) -> np.ndarray:
        """n×n，M[i,j] = P(i 赢 j)。只算上半三角，下半取补。"""
        n = len(teams)
        M = np.full((n, n), 0.5)
        todo = [(i, j) for i in range(n) for j in range(i + 1, n)]
        t0 = time.time()
        for done, (i, j) in enumerate(todo, 1):
            M[i, j] = self.probs(teams[i], teams[j], (fmt,))[fmt]
            M[j, i] = 1.0 - M[i, j]
            if progress and self.verbose and done % 30 == 0:
                print(f"  [{fmt}] {done}/{len(todo)} 组 {time.time() - t0:.0f}s", flush=True)
        return M


# ================================================================ 瑞士轮模拟
_MASK_BITS = tuple(tuple(i for i in range(N_TEAMS) if (m >> i) & 1) for m in range(1 << N_TEAMS))


def _greedy_pairs(order: list[int], mask: list[int]) -> list[tuple[int, int]]:
    """第 2/3 轮：组内最高种子 vs 最低的未交手种子。"""
    pool = list(order)
    pairs: list[tuple[int, int]] = []
    while pool:
        a = pool.pop(0)
        k = len(pool) - 1
        while k >= 0 and (mask[a] >> pool[k]) & 1:
            k -= 1
        pairs.append((a, pool.pop(k if k >= 0 else len(pool) - 1)))
    return pairs


def _all_matchings(seq: list[int]):
    """枚举完美匹配，优先让前一个位置配最靠后的（即高种子对低种子）。"""
    if not seq:
        yield []
        return
    first, rest = seq[0], seq[1:]
    for k in range(len(rest) - 1, -1, -1):
        for tail in _all_matchings(rest[:k] + rest[k + 1:]):
            yield [(first, rest[k])] + tail


def _matchings_without_rematch(order: list[int], mask: list[int]) -> list[tuple[int, int]]:
    """贪心撞上重赛时才调用：在全部完美匹配里找第一条不重赛的（否则取重赛最少）。"""
    best, best_rem = None, None
    for pairs in _all_matchings(list(order)):
        rem = sum(1 for i, j in pairs if (mask[i] >> j) & 1)
        if rem == 0:
            return pairs
        if best_rem is None or rem < best_rem:
            best, best_rem = pairs, rem
    return best if best is not None else _greedy_pairs(order, mask)


def simulate(ids: np.ndarray, p_bo1: np.ndarray, p_bo3: np.ndarray, seed: int = 0,
             all_bo3: bool = False) -> dict:
    """按 Valve 规则跑 n 次完整 16 队瑞士轮。

    赛制由规则决定，不是统一胜率：**晋级战与淘汰战用 BO3 概率，其余用 BO1 概率**。
    等价判据：任一方已 2 胜（能晋级）或已 2 负（会被淘汰）即为 BO3。
    16 队五轮下来恒定 20 场 BO1 + 13 场 BO3：
      第1轮 0-0 池 8 场 BO1；第2轮 1-0/0-1 池 8 场 BO1；
      第3轮 2-0 池 2 场 BO3、1-1 池 4 场 BO1、0-2 池 2 场 BO3；
      第4轮 2-1 池 3 场 BO3、1-2 池 3 场 BO3；第5轮 2-2 池 3 场 BO3。
    这与 PGL/Valve 的日程一致（首日全 BO1、次日混编、第三四日全 BO3）。

    all_bo3=True 时覆盖默认赛制，全部 33 场使用 BO3 胜率。
    ids: (n,16) 全局队伍号，按本阶段种子从高到低；单阶段时各行相同。
    """
    n = ids.shape[0]
    P1, P3 = p_bo1.tolist(), p_bo3.tolist()
    urandom = random.Random(seed).random
    bits = _MASK_BITS
    wins = np.zeros((n, N_TEAMS), np.int8)
    losses = np.zeros((n, N_TEAMS), np.int8)
    diff = np.zeros((n, N_TEAMS), np.int16)
    fallback = 0
    tally = [0, 0, 0, 0]     # [BO1 场数, BO3 场数, 重赛次数, 贪心改判次数]

    for s in range(n):
        row = ids[s]
        w = [0] * N_TEAMS
        l = [0] * N_TEAMS
        mask = [0] * N_TEAMS

        def play(i: int, j: int) -> None:
            bo3 = all_bo3 or w[i] == 2 or w[j] == 2 or l[i] == 2 or l[j] == 2
            p = (P3 if bo3 else P1)[row[i]][row[j]]
            tally[1 if bo3 else 0] += 1
            if (mask[i] >> j) & 1:          # 同阶段内重赛（应恒为 0）
                tally[2] += 1
            mask[i] |= 1 << j
            mask[j] |= 1 << i
            if urandom() < p:
                w[i] += 1
                l[j] += 1
            else:
                w[j] += 1
                l[i] += 1

        for i, j in INITIAL_PAIRS:
            play(i, j)

        for round_no in (2, 3, 4, 5):
            groups: dict[tuple[int, int], list[int]] = {}
            for t in range(N_TEAMS):
                if w[t] >= 3 or l[t] >= 3:
                    continue
                groups.setdefault((w[t], l[t]), []).append(t)
            for members in groups.values():
                if len(members) < 2:
                    continue
                dif = {t: sum(w[o] - l[o] for o in bits[mask[t]]) for t in members}
                order = sorted(members, key=lambda t: (-dif[t], t))
                if round_no <= 3 or len(order) != 6:
                    # 第 2/3 轮：组内最高种子 vs 最低的未交手种子
                    pairs = _greedy_pairs(order, mask)
                    if any((mask[i] >> j) & 1 for i, j in pairs):
                        # 贪心撞车但换一种配对可以完全避开重赛（规则要求 if possible）
                        tally[3] += 1
                        pairs = _matchings_without_rematch(order, mask)
                else:
                    # 第 4/5 轮：15 行优先级表取第一条不重赛的行
                    pairs = None
                    for row_pairs in PRIORITY_ROWS:
                        if all(not (mask[order[p]] >> order[q]) & 1 for p, q in row_pairs):
                            pairs = [(order[p], order[q]) for p, q in row_pairs]
                            break
                    if pairs is None:
                        fallback += 1
                        best = min(PRIORITY_ROWS,
                                   key=lambda rp: sum((mask[order[p]] >> order[q]) & 1 for p, q in rp))
                        pairs = [(order[p], order[q]) for p, q in best]
                for i, j in pairs:
                    play(i, j)

        for t in range(N_TEAMS):
            wins[s, t] = w[t]
            losses[s, t] = l[t]
            diff[s, t] = sum(w[o] - l[o] for o in bits[mask[t]])

    code = np.zeros((n, N_TEAMS), np.int8)
    for (wi, li), ci in CODE_OF.items():
        code[(wins == wi) & (losses == li)] = ci
    return dict(wins=wins, losses=losses, difficulty=diff, code=code, fallback=fallback,
                bo1_matches=tally[0] / n, bo3_matches=tally[1] / n, rematches=tally[2],
                all_bo3=all_bo3)


def check_structure(code: np.ndarray) -> dict[str, int]:
    """核对结果分布是否恒为 2/3/3/3/3/2 —— 这是规则的结构性事实，用来抓配对逻辑的 bug。"""
    counts = np.bincount(code.ravel(), minlength=6)
    if not np.array_equal(counts, STRUCTURAL_COUNTS * code.shape[0]):
        raise AssertionError(f"瑞士轮结果分布异常：{counts.tolist()}")
    return {name: int(c) for name, c in zip(OUTCOMES, counts)}


def final_order(code: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """阶段内最终种子序：（结果编码, −Difficulty, 初始种子）。返回 (n,16) 的队伍列号。"""
    key = (-code.astype(np.int32) * 100000 + diff.astype(np.int32) * 1000
           - np.arange(code.shape[1]))
    return np.argsort(-key, axis=1, kind="stable")


# ================================================================ Pick'Em 优化
SLOT_3_0, SLOT_ADV, SLOT_0_3 = 1, 2, 3        # 标签；0 = 未选
SLOT_CAP = {SLOT_3_0: 2, SLOT_ADV: 6, SLOT_0_3: 2}
SLOT_INDEX = {SLOT_3_0: 0, SLOT_ADV: 1, SLOT_0_3: 2}


def build_indicators(code: np.ndarray) -> np.ndarray:
    """(n,16,3) uint8：某队放在某槽位时该格是否命中。"""
    return np.stack([code == 0, (code == 1) | (code == 2), code == 5], axis=2).astype(np.uint8)


class PickEm:
    """在模拟出的联合分布上求 P(答对 ≥ K) 最大的 10 格分配。

    目标不是"各格边际之和"（那只是最大化期望答对数）：10 格之间强相关
    （恰好 2 支队 3-0、恰好 2 支队 0-3、恰好 6 支队 3-1/3-2），
    "期望答对数最大"与"通过概率最大"不是同一个最优点，所以直接在计数上优化。
    """

    def __init__(self, code: np.ndarray):
        self.code = code
        self.n = code.shape[0]
        self.I = build_indicators(code)
        self.I16 = self.I.astype(np.int16)
        self.marginal = self.I.mean(axis=0)          # (16,3)

    def score_of(self, labels: np.ndarray) -> np.ndarray:
        s = np.zeros(self.n, np.int16)
        for slot, col in SLOT_INDEX.items():
            chosen = np.nonzero(labels == slot)[0]
            if chosen.size:
                s += self.I16[:, chosen, col].sum(axis=1, dtype=np.int16)
        return s

    def stats(self, labels: np.ndarray, thresholds: list[int]) -> dict:
        s = self.score_of(labels)
        hist = np.bincount(s, minlength=11)[:11]
        return dict(
            distribution={str(k): int(hist[k]) for k in range(11)},
            probability={str(k): float(hist[k] / self.n) for k in range(11)},
            expected=float(s.mean()),
            at_least={str(k): float(hist[k:].sum() / self.n) for k in thresholds},
            labels=list(map(int, labels)),
        )

    def greedy_expected(self) -> np.ndarray:
        """最大化期望答对数：16 队 × 24 列的指派问题（2 个 3-0、6 个晋级、2 个 0-3、14 个空位）。"""
        from scipy.optimize import linear_sum_assignment
        cols = [SLOT_3_0] * 2 + [SLOT_ADV] * 6 + [SLOT_0_3] * 2 + [0] * (N_TEAMS - 10)
        cost = np.zeros((N_TEAMS, len(cols)))
        for c, slot in enumerate(cols):
            if slot:
                cost[:, c] = -self.marginal[:, SLOT_INDEX[slot]]
        rows, assigned = linear_sum_assignment(cost)
        labels = np.zeros(N_TEAMS, np.int8)
        for r, c in zip(rows, assigned):
            labels[r] = cols[c]
        return labels

    def _swap_scores(self, sv: np.ndarray, i: int, j: int, li: int, lj: int) -> np.ndarray:
        out = sv
        if li:
            out = out - self.I16[:, i, SLOT_INDEX[li]]
        if lj:
            out = out - self.I16[:, j, SLOT_INDEX[lj]]
        if lj:
            out = out + self.I16[:, i, SLOT_INDEX[lj]]
        if li:
            out = out + self.I16[:, j, SLOT_INDEX[li]]
        return out

    def local_search(self, labels: np.ndarray, k: int, max_passes: int = 50) -> tuple[np.ndarray, np.ndarray]:
        sv = self.score_of(labels)
        best = float(np.count_nonzero(sv >= k))
        for _ in range(max_passes):
            improved = False
            for i in range(N_TEAMS):
                for j in range(i + 1, N_TEAMS):
                    li, lj = int(labels[i]), int(labels[j])
                    if li == lj:
                        continue
                    cand = self._swap_scores(sv, i, j, li, lj)
                    score = float(np.count_nonzero(cand >= k))
                    if score > best or (score == best and cand.sum() > sv.sum()):
                        labels = labels.copy()
                        labels[i], labels[j] = lj, li
                        sv, best, improved = cand, score, True
            if not improved:
                break
        return labels, sv

    def kick(self, labels: np.ndarray, rng: random.Random, swaps: int = 2) -> np.ndarray:
        """扰动：随机交换若干对格子的标签（交换永远保持 2/6/2 配额合法）。"""
        out = labels.copy()
        for _ in range(swaps):
            i, j = rng.sample(range(N_TEAMS), 2)
            out[i], out[j] = out[j], out[i]
        return out

    def solve(self, k: int, restarts: int = 8, kicks: int = 40,
              rng: random.Random | None = None) -> dict:
        """贪心起点 + 随机起点 → 1-swap 局部搜索 → 迭代局部搜索（扰动后重搜）。

        1-swap 邻域是**穷举**的（每轮把全部 C(16,2)=120 对都试一遍），所以返回的解是
        真正的 1-swap 局部最优；kick 用来跳出局部最优。仍然不保证全局最优。
        """
        rng = rng or random.Random(0)
        starts = [self.greedy_expected()]
        slots = [SLOT_3_0] * 2 + [SLOT_ADV] * 6 + [SLOT_0_3] * 2
        for _ in range(restarts):
            labels = np.zeros(N_TEAMS, np.int8)
            picked = rng.sample(range(N_TEAMS), 10)
            rng.shuffle(slots)
            for t, slot in zip(picked, slots):
                labels[t] = slot
            starts.append(labels)
        best = None
        for labels in starts:
            labels, sv = self.local_search(labels, k)
            prob, expected = float(np.count_nonzero(sv >= k)) / self.n, float(sv.mean())
            if best is None or (prob, expected) > (best["probability"], best["expected"]) + (1e-12, 1e-12):
                best = dict(labels=labels, k=k, probability=prob, expected=expected)
        for _ in range(kicks):
            cand = self.kick(best["labels"], rng, swaps=rng.choice((2, 3)))
            cand, sv = self.local_search(cand, k)
            prob, expected = float(np.count_nonzero(sv >= k)) / self.n, float(sv.mean())
            if (prob, expected) > (best["probability"], best["expected"]) + (1e-12, 1e-12):
                best = dict(labels=cand, k=k, probability=prob, expected=expected)
        return best


def describe_pick(labels: np.ndarray, teams: list[str], marginal: np.ndarray) -> dict:
    out = {"3-0": [], "晋级(3-1/3-2)": [], "0-3": []}
    for slot, name in ((SLOT_3_0, "3-0"), (SLOT_ADV, "晋级(3-1/3-2)"), (SLOT_0_3, "0-3")):
        col = SLOT_INDEX[slot]
        for t in np.nonzero(labels == slot)[0]:
            out[name].append(dict(team=teams[t], seed=int(t) + 1, hit=float(marginal[t, col])))
    return out


# ================================================================ 报告
def _pick_text(items: list[dict]) -> str:
    return "、".join(f"{d['team']} {d['hit']:.0%}" for d in items)


def rarity_distribution(pick: "PickEm", ks: list[int], samples: int = 1000, seed: int = 7) -> dict:
    """从全部合法作业里随机抽样本，给出"随便填"能拿到多少通过率的基准分布。"""
    rng = random.Random(seed)
    slots = [SLOT_3_0] * 2 + [SLOT_ADV] * 6 + [SLOT_0_3] * 2
    out = {k: [] for k in ks}
    for _ in range(samples):
        labels = np.zeros(N_TEAMS, np.int8)
        picked = rng.sample(range(N_TEAMS), 10)
        rng.shuffle(slots)
        for t, slot in zip(picked, slots):
            labels[t] = slot
        st = pick.stats(labels, ks)
        for k in ks:
            out[k].append(st["at_least"][str(k)])
    return {k: np.asarray(v) for k, v in out.items()}


def independent_baseline(pick: "PickEm", labels: np.ndarray, k: int) -> float:
    """错口径对照：若把 10 格当成相互独立、用各自边际做 Poisson-binomial。

    真实答对数不是这样——10 格强相关（正好 2 支队 3-0、正好 6 支 3-1/3-2、正好 2 支 0-3，
    而且被填进 3-0 格却打成 3-1 的队，在晋级格里同样不得分）。这个数只用来量化"忽略联合"的代价。
    """
    ps = []
    for slot, col in SLOT_INDEX.items():
        for t in np.nonzero(labels == slot)[0]:
            ps.append(float(pick.marginal[t, col]))
    dp = np.zeros(len(ps) + 1)
    dp[0] = 1.0
    for p in ps:
        dp[1:] = dp[1:] * (1 - p) + dp[:-1] * p
        dp[0] *= (1 - p)
    return float(dp[k:].sum())


def holdout_check(pick: "PickEm", k: int, cap: int = 60000,
                  rng: random.Random | None = None) -> dict:
    """半样本交叉验证：在一半样本上求解、另一半上看成绩；并量化"挑样本内最好"的选择偏差。"""
    rng = rng or random.Random(99)
    step = max(1, pick.n // cap)
    idx = np.arange(0, pick.n, step)
    even, odd = idx[0::2], idx[1::2]
    A, B = PickEm(pick.code[even]), PickEm(pick.code[odd])
    solA = A.solve(k, restarts=4, kicks=12, rng=random.Random(rng.randint(0, 10 ** 6)))
    solB = B.solve(k, restarts=4, kicks=12, rng=random.Random(rng.randint(0, 10 ** 6)))
    out = dict(half_n=int(even.size),
               in_a=float(solA["probability"]),
               oos_a=float(B.stats(solA["labels"], [k])["at_least"][str(k)]),
               in_b=float(solB["probability"]),
               oos_b=float(A.stats(solB["labels"], [k])["at_least"][str(k)]),
               labels_a=[int(v) for v in solA["labels"]],
               labels_b=[int(v) for v in solB["labels"]],
               same=bool(np.array_equal(solA["labels"], solB["labels"])))

    # 选择偏差：在 A 上从随机作业里挑出最好的，看它在 B 上还剩多少
    slots = [SLOT_3_0] * 2 + [SLOT_ADV] * 6 + [SLOT_0_3] * 2
    best_labels, best_in = None, -1.0
    for _ in range(600):
        labels = np.zeros(N_TEAMS, np.int8)
        picked = rng.sample(range(N_TEAMS), 10)
        rng.shuffle(slots)
        for t, slot in zip(picked, slots):
            labels[t] = slot
        v = float(A.stats(labels, [k])["at_least"][str(k)])
        if v > best_in:
            best_in, best_labels = v, labels
    out.update(random_best_in=float(best_in),
               random_best_oos=float(B.stats(best_labels, [k])["at_least"][str(k)]))
    return out


def stage_report(pw: Pairwise, teams: list[str], sim: dict, pick: PickEm, search: PickEm,
                 thresholds: list[int], sims: int, seed: int, notes: list[str]) -> tuple[str, dict]:
    code = sim["code"]
    n = code.shape[0]
    dist = np.stack([(code == c).mean(axis=0) for c in range(6)], axis=1)      # (16,6)
    advance = dist[:, :3].sum(axis=1)
    exp_wins = dist @ np.array([3, 3, 3, 2, 1, 0])

    solves = {k: search.solve(k, restarts=8, rng=random.Random(seed + k)) for k in thresholds}
    for sol in solves.values():                       # 在全部模拟样本上复核
        sol["labels"] = np.asarray(sol["labels"], np.int8)
        full = pick.stats(sol["labels"], thresholds)
        sol.update(full=full, probability=full["at_least"][str(sol["k"])])
    greedy = pick.greedy_expected()
    chalk = np.zeros(N_TEAMS, np.int8)
    chalk[0:2], chalk[2:8], chalk[14:16] = SLOT_3_0, SLOT_ADV, SLOT_0_3

    L = ["# Major 瑞士轮模拟与 Pick'Em 作业", ""]
    L += [f"- {x}" for x in notes]
    L += [f"- 模拟次数 {sims:,}；随机种子 {seed}；两两胜率引擎 {pw.engine}，"
          f"对称护栏 [{pw.rail:.2f}, {1 - pw.rail:.2f}]，as_of {pw.as_of}，"
          f"建模截止 {pw.model['trained_through']}",
          f"- 第 4/5 轮找不到「全不重赛」行的次数：{sim['fallback']}（应为 0）", ""]

    L += ["## 1. 本模拟采用的 Valve 瑞士轮规则", "",
          "| 项 | 规则 |", "|---|---|",
          "| 阶段 | Stage 1/2/3 各 16 队瑞士轮、五轮；三胜晋级、三负淘汰 |",
          "| 配对 | 同战绩对同战绩；第 1 轮固定 1v9…8v16；第 2/3 轮组内「最高种子 vs 最低的未交手种子」；"
          "第 4/5 轮按 15 行优先级表取第一条不重赛的行 |",
          ("| 赛制 | **全部 BO3（--all-bo3）** |" if sim.get("all_bo3", False)
           else "| 赛制 | **晋级战与淘汰战 BO3，其余全部 BO1** |"),
          "| 阶段内种子 |（战绩，Difficulty，初始种子）；Difficulty = 已交手对手（胜−负）之和 |",
          "| 阶段 2/3 种子 | 直邀 8 队 = 1-8 号（赛前 VRS 序）；上一阶段晋级 8 队 = 9-16 号（上一阶段最终种子序） |",
          "",
          "### 1.1 哪些场是 BO1、哪些是 BO3", "",
          "判据（直接来自规则书）：**任一方已 2 胜（赢下即晋级）或已 2 负（输掉即淘汰）→ BO3，其余 BO1**。", "",
          "| 轮次 | 对阵池 | 场数 | 赛制 | 说明 |", "|---|---|---:|---|---|",
          "| 第 1 轮 | 0-0 | 8 | BO1 | 固定 1v9、2v10 … 8v16 |",
          "| 第 2 轮 | 1-0 / 0-1 | 4 + 4 | BO1 | 组内最高种子 vs 最低的未交手种子 |",
          "| 第 3 轮 | **2-0** | 2 | **BO3** | 胜者 3-0 晋级 |",
          "| | 1-1 | 4 | BO1 | |",
          "| | **0-2** | 2 | **BO3** | 败者 0-3 淘汰 |",
          "| 第 4 轮 | **2-1** | 3 | **BO3** | 胜者 3-1 晋级 |",
          "| | **1-2** | 3 | **BO3** | 败者 1-3 淘汰 |",
          "| 第 5 轮 | **2-2** | 3 | **BO3** | 胜者晋级、败者淘汰 |",
          f"| 合计 | | 33 | **20 BO1 + 13 BO3** | 本次模拟实测 {sim['bo1_matches']:.0f} + "
          f"{sim['bo3_matches']:.0f} 场 |", "",
          "BO1 与 BO3 各用各的胜率：同一套逐图 p，走不同的 veto 协议与系列赛组合，"
          "不是把 BO1 胜率平方一次了事。", "",
          "由规则推出的结构事实：每阶段恒定 2 个 3-0、3 个 3-1、3 个 3-2、3 个 2-3、"
          "3 个 1-3、2 个 0-3。", ""]

    if sim.get("all_bo3", False):
        L[L.index("## 1. 本模拟采用的 Valve 瑞士轮规则")] = "## 1. 本模拟采用的瑞士轮规则（全 BO3 覆盖）"
        start = L.index("### 1.1 哪些场是 BO1、哪些是 BO3")
        end = next(i for i in range(start, len(L)) if L[i].startswith("由规则推出的结构事实"))
        L[start:end] = ["### 1.1 全 BO3 模式", "",
                        "本阶段启用 `--all-bo3`：全部五轮均使用 BO3 胜率，共 0 场 BO1 + 33 场 BO3。",
                        "BO3 胜率由逐图概率与 BO3 veto 协议计算；配对和晋级淘汰规则保持不变。", ""]

    L += ["## 2. 每队最终战绩概率", "",
          "按晋级概率降序。`3-0/3-1/3-2` 晋级，`2-3/1-3/0-3` 淘汰。", "",
          "| 种子 | 队伍 | 3-0 | 3-1 | 3-2 | 2-3 | 1-3 | 0-3 | 晋级 | 期望胜场 |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in np.argsort(-advance):
        L.append(f"| {t + 1} | {teams[t]} | " + " | ".join(f"{dist[t, c]:.1%}" for c in range(6))
                 + f" | **{advance[t]:.1%}** | {exp_wins[t]:.2f} |")
    L.append("")

    L += ["## 3. 每格候选（边际命中率）", "",
          "「命中」= 该队放在该槽位时答对的概率。3-0 与 0-3 各只有 2 支队能答对，"
          "所以单格命中率天然很低；晋级格的结构上限是 6/16。", ""]
    for slot, name in ((SLOT_3_0, "3-0 格（选 2 个）"), (SLOT_ADV, "晋级格 3-1/3-2（选 6 个）"),
                       (SLOT_0_3, "0-3 格（选 2 个）")):
        col = SLOT_INDEX[slot]
        ranked = np.argsort(-pick.marginal[:, col])
        L += [f"**{name}** — 边际前 8：" + "、".join(
            f"{teams[t]} {pick.marginal[t, col]:.1%}" for t in ranked[:8]), ""]

    L += ["## 4. 推荐作业", "",
          f"**通过线**：答对 ≥{max(thresholds)} 个才算过。下面是使通过概率最大的 10 格，"
          "括号内是该格命中的概率。", ""]
    for k in thresholds:
        sol = solves[k]
        d = describe_pick(sol["labels"], teams, pick.marginal)
        st = sol["full"]
        L += [f"### 目标：答对 ≥ {k} 个 → 最优命中率 **{st['at_least'][str(k)]:.1%}**", "",
              f"- **3-0**：{_pick_text(d['3-0'])}",
              f"- **晋级 3-1/3-2**：{_pick_text(d['晋级(3-1/3-2)'])}",
              f"- **0-3**：{_pick_text(d['0-3'])}",
              "- 答对数分布：" + "，".join(
                  f"{i} 个 {st['probability'][str(i)]:.1%}" for i in range(11)
                  if st['probability'][str(i)] > 5e-4),
              f"- 期望答对 {st['expected']:.2f} 个；" + "；".join(
                  f"≥{kk} 个 {st['at_least'][str(kk)]:.1%}" for kk in thresholds), ""]

    for name, labels, why in (
            ("只最大化期望答对数", greedy, "不看通过率，只看平均答对几个"),
            ("按种子硬填", chalk, "2 个最强 3-0、6 个次强晋级、2 个最弱 0-3")):
        st = pick.stats(labels, thresholds)
        d = describe_pick(labels, teams, pick.marginal)
        L += [f"### 对照：{name}（{why}）", "",
              f"- 3-0：{_pick_text(d['3-0'])}；晋级：{_pick_text(d['晋级(3-1/3-2)'])}；"
              f"0-3：{_pick_text(d['0-3'])}",
              f"- 期望答对 {st['expected']:.2f} 个；" + "；".join(
                  f"≥{kk} 个 {st['at_least'][str(kk)]:.1%}" for kk in thresholds), ""]

    L += ["## 5. 推荐解到底有多稀罕", "",
          "从全部合法作业里随机抽样本，看「随便填」能拿到多少，推荐解站在什么位置：", "",
          "| 目标 | 随机 5% 分位 | 随机中位 | 随机 95% 分位 | 随机最好 | 推荐解 | 推荐解分位 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    rand = rarity_distribution(search, thresholds)
    for k in thresholds:
        v = rand[k]
        rec = solves[k]["probability"]
        L.append(f"| ≥{k} | {np.quantile(v, .05):.1%} | {np.median(v):.1%} | "
                 f"{np.quantile(v, .95):.1%} | {v.max():.1%} | **{rec:.1%}** | "
                 f"{float((v <= rec).mean()):.0%} 分位 |")
    L += ["", f"同一目标下候选之间只差几个百分点时属于抽样噪声（本次 {sims:,} 次模拟，"
              f"单点通过率标准误约 {math.sqrt(.35 * .65 / sims):.2%}），不必纠结换哪一格。", ""]

    k_main = max(thresholds)
    hold = holdout_check(pick, k_main, rng=random.Random(seed + 11))
    naive = independent_baseline(pick, solves[k_main]["labels"], k_main)
    L += ["## 6. 这个概率是怎么来的、有多可信", "",
          f"`P(≥{k_main})` = **{sims:,} 条完整瑞士轮模拟中，这套 10 格答对 ≥{k_main} 个的比例**。"
          "每条模拟给出一个完整的 16 队最终战绩联合结果（2 个 3-0、3 个 3-1、3 个 3-2、"
          "3 个 2-3、3 个 1-3、2 个 0-3），逐条数命中格数后取平均；评分阶段**没有任何独立性假设**，"
          "也不从边际反推。真正需要检查的是「在同一批样本上挑最优」带来的偏差，所以：", "",
          "**① 半样本交叉验证**（一半样本上求解，另一半样本上看成绩）：", "",
          "| 训练半 → 检验半 | 样本内 | 样本外 |", "|---|---:|---:|",
          f"| 偶数样本求解 → 奇数样本检验 | {hold['in_a']:.1%} | {hold['oos_a']:.1%} |",
          f"| 奇数样本求解 → 偶数样本检验 | {hold['in_b']:.1%} | {hold['oos_b']:.1%} |",
          f"| 两半是否选出同一套作业 | {'是' if hold['same'] else '否'} | — |", "",
          f"**② 选择偏差对照**：在 600 个随机作业里挑「样本内最好」的那个，它在另一半样本上从 "
          f"{hold['random_best_in']:.1%} 变成 {hold['random_best_oos']:.1%}"
          f"（样本外 − 样本内 = {(hold['random_best_oos'] - hold['random_best_in']) * 100:+.1f}pp）。"
          "这就是「在同一样本上挑最好」的平均虚高幅度。", "",
          f"**③ 若错误地把 10 格当独立**（用各格边际做 Poisson-binomial）会得到 {naive:.1%}，"
          f"与真实的 {solves[k_main]['full']['at_least'][str(k_main)]:.1%} 相差 "
          f"{abs(naive - solves[k_main]['full']['at_least'][str(k_main)]) * 100:.1f}pp ——"
          "这就是忽略联合（槽位互斥、3-0 队填进晋级格不得分）的代价。", ""]

    L += ["## 7. 两两胜率矩阵（BO1 / BO3）", "",
          "行 = 左边队伍，列 = 右边队伍；值是该行队伍获胜的概率。只作核对用。", "",
          "| # | 队伍 | " + " | ".join(str(i + 1) for i in range(N_TEAMS)) + " |",
          "|---:|---|" + "---|" * N_TEAMS]
    for i in range(N_TEAMS):
        cells = []
        for j in range(N_TEAMS):
            if i == j:
                cells.append("—")
            elif i < j:
                p = pw.probs(teams[i], teams[j])
                cells.append(f"{p['BO1']:.2f}/{p['BO3']:.2f}")
            else:
                p = pw.probs(teams[j], teams[i])
                cells.append(f"{1 - p['BO1']:.2f}/{1 - p['BO3']:.2f}")
        L.append(f"| {i + 1} | {teams[i]} | " + " | ".join(cells) + " |")
    L.append("")

    payload = dict(
        teams=teams, sims=sims, seed=seed, as_of=pw.as_of, engine=pw.engine, rail=pw.rail,
        fingerprint=pw.fingerprint, trained_through=pw.model["trained_through"],
        structure=check_structure(code), fallback=sim["fallback"],
        bo1_matches=sim["bo1_matches"], bo3_matches=sim["bo3_matches"],
        all_bo3=sim.get("all_bo3", False),
        rarity={str(k): dict(p05=float(np.quantile(rand[k], .05)), median=float(np.median(rand[k])),
                             p95=float(np.quantile(rand[k], .95)), best=float(rand[k].max()))
                for k in thresholds},
        per_team=[dict(seed=t + 1, team=teams[t],
                       p={OUTCOMES[c]: float(dist[t, c]) for c in range(6)},
                       p_advance=float(advance[t]), expected_wins=float(exp_wins[t]),
                       marginal={"3-0": float(pick.marginal[t, 0]),
                                 "advance": float(pick.marginal[t, 1]),
                                 "0-3": float(pick.marginal[t, 2])}) for t in range(N_TEAMS)],
        solutions={str(k): dict(probability=float(v["probability"]), stats=v["full"],
                                picks=describe_pick(v["labels"], teams, pick.marginal))
                   for k, v in solves.items()},
        baseline_expectation=dict(stats=pick.stats(greedy, thresholds),
                                  picks=describe_pick(greedy, teams, pick.marginal)),
        baseline_chalk=dict(stats=pick.stats(chalk, thresholds),
                            picks=describe_pick(chalk, teams, pick.marginal)),
        validation=dict(holdout=hold, independent_baseline=float(naive),
                        note="holdout = 半样本交叉验证；independent_baseline = 错误地按独立格计算的对照"),
    )
    return "\n".join(L) + "\n", payload


# ================================================================ 主流程
def _write_outputs(out_dir: Path, md: str, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "prediction.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    with (out_dir / "outcomes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seed", "team", *OUTCOMES, "advance", "expected_wins"])
        for row in payload["per_team"]:
            wr.writerow([row["seed"], row["team"], *[f"{row['p'][o]:.6f}" for o in OUTCOMES],
                         f"{row['p_advance']:.6f}", f"{row['expected_wins']:.4f}"])


def _stage_teams(pw: Pairwise, display: list[str]) -> list[str]:
    if len(display) not in (8, N_TEAMS):
        raise ValueError(f"种子表需要 8 支（只填直邀）或 16 支队伍，当前 {len(display)} 支")
    teams = [pw.resolve(t) for t in display]
    dup = sorted({pw.names[t] for t in teams if teams.count(t) > 1})
    if dup:
        raise ValueError(f"种子表里有重复队伍：{dup}")
    return teams


def _simulate_stage(pw: Pairwise, teams: list[str], sims: int, seed: int,
                    all_bo3: bool = False) -> dict:
    M1 = pw.matrix(teams, "BO1")
    M3 = pw.matrix(teams, "BO3")
    ids = np.tile(np.arange(len(teams), dtype=np.int16), (sims, 1))
    return simulate(ids, M1, M3, seed=seed, all_bo3=all_bo3)


def _stage_payload(pw: Pairwise, teams: list[str], sim: dict, sims: int, seed: int,
                   thresholds: list[int], title: str) -> tuple[str, dict, dict]:
    structure = check_structure(sim["code"])
    expected = (0.0, 33.0) if sim.get("all_bo3", False) else (20.0, 13.0)
    if (sim["bo1_matches"], sim["bo3_matches"]) != expected:
        raise AssertionError(f"BO1/BO3 场数异常：{sim['bo1_matches']}/{sim['bo3_matches']}（应为 {expected}）")
    pick = PickEm(sim["code"])
    step = max(1, sims // 30000)
    search = PickEm(sim["code"][::step]) if step > 1 else pick
    md, payload = stage_report(pw, [pw.names[t] for t in teams], sim, pick, search,
                               thresholds, sims, seed, [title, EVENT_NOTE])
    return md, payload, structure


def run_stage(pw: Pairwise, teams_display: list[str], sims: int, seed: int,
              thresholds: list[int], out_dir: Path, title: str, all_bo3: bool = False) -> dict:
    if len(teams_display) != N_TEAMS:
        raise ValueError(
            f"种子表需要 {N_TEAMS} 支队伍，当前 {len(teams_display)} 支。"
            "Stage 2/3 的 9-16 号种子要等上一阶段打完才能填；"
            "想先看整届走势用 --major S1 S2 S3（会自动补齐）。")
    teams = _stage_teams(pw, teams_display)
    print(f"[对阵] {len(teams)} 队两两 BO1 / BO3 胜率…", flush=True)
    t0 = time.time()
    M1 = pw.matrix(teams, "BO1")
    M3 = pw.matrix(teams, "BO3")
    pw.save_cache()

    ids = np.tile(np.arange(N_TEAMS, dtype=np.int16), (sims, 1))
    print(f"[模拟] {sims:,} 次…", flush=True)
    sim = simulate(ids, M1, M3, seed=seed, all_bo3=all_bo3)
    md, payload, structure = _stage_payload(pw, teams, sim, sims, seed, thresholds, title)
    print(f"[模拟] {time.time() - t0:.1f}s  BO1 {sim['bo1_matches']:.0f} 场 + BO3 "
          f"{sim['bo3_matches']:.0f} 场  结构分布 {structure}", flush=True)
    _write_outputs(out_dir, md, payload)
    return payload


# ---------------------------------------------------------------- 整届 Major
def advancer_ids(ids: np.ndarray, sim: dict, k: int = 8) -> np.ndarray:
    """按阶段内最终种子序，取前 k 名的全局队伍号。"""
    order = final_order(sim["code"], sim["difficulty"])[:, :k]
    return np.take_along_axis(ids, order, axis=1)


def run_playoffs(ids: np.ndarray, sim: dict, M3: np.ndarray, seed: int) -> dict:
    """8 队单败季后赛：1v8 4v5 / 2v7 3v6，全部 BO3（规则书）。"""
    order = final_order(sim["code"], sim["difficulty"])[:, :8]
    seeds = np.take_along_axis(ids, order, axis=1)      # (n,8) 全局队伍号，1..8 号种子
    n = seeds.shape[0]
    P3 = M3.tolist()
    urandom = random.Random(seed + 7707).random
    stage = np.zeros((n, 8), np.int8)                   # 1=四强 2=决赛 3=夺冠

    for s in range(n):
        row = seeds[s].tolist()
        winners = []
        for a, b in ((0, 7), (3, 4), (1, 6), (2, 5)):
            x, y = row[a], row[b]
            win_a = urandom() < P3[x][y]
            if win_a:
                winners.append((x, a))
                stage[s, b] = 0
            else:
                winners.append((y, b))
        for (t, pos) in winners:
            stage[s, pos] = 1
        finalists = []
        for (x, px), (y, py) in ((winners[0], winners[1]), (winners[2], winners[3])):
            if urandom() < P3[x][y]:
                finalists.append((x, px))
                stage[s, py] = 1
            else:
                finalists.append((y, py))
                stage[s, px] = 1
        for t, p in finalists:
            stage[s, p] = 2
        (x, px), (y, py) = finalists
        if urandom() < P3[x][y]:
            stage[s, px] = 3
        else:
            stage[s, py] = 3
    return dict(seeds=seeds, stage=stage)


def run_major(pw: Pairwise, paths: list[Path], sims: int, seed: int,
              thresholds: list[int], out_dir: Path, stage3_all_bo3: bool = False) -> dict:
    displays = [read_seeds(p) for p in paths]
    stages = [_stage_teams(pw, d) for d in displays]
    universe = list(dict.fromkeys(t for st in stages for t in st))
    gid = {t: i for i, t in enumerate(universe)}
    print(f"[对阵] 全届 {len(universe)} 队两两 BO1 / BO3 胜率…", flush=True)
    t0 = time.time()
    M1 = pw.matrix(universe, "BO1")
    M3 = pw.matrix(universe, "BO3")
    pw.save_cache()
    print(f"[对阵] {time.time() - t0:.0f}s", flush=True)

    ids = np.tile(np.array([gid[t] for t in stages[0]], np.int16), (sims, 1))
    sims_by_stage, fields = [], []
    adv = None
    for k, st in enumerate(stages, 1):
        if k > 1:
            invited = np.array([gid[t] for t in st], np.int16)
            if len(st) == 8:
                ids = np.hstack([np.tile(invited, (sims, 1)), adv])
            else:
                ids = np.tile(invited, (sims, 1))
        print(f"[模拟] Stage {k}：{sims:,} 次…", flush=True)
        all_bo3 = stage3_all_bo3 and k == 3
        sim = simulate(ids, M1, M3, seed=seed + k, all_bo3=all_bo3)
        structure = check_structure(sim["code"])
        expected = (0.0, 33.0) if all_bo3 else (20.0, 13.0)
        if (sim["bo1_matches"], sim["bo3_matches"]) != expected:
            raise AssertionError(f"Stage {k} 的 BO1/BO3 场数异常")
        print(f"[模拟] Stage {k} 完成  结构分布 {structure}", flush=True)
        sims_by_stage.append(sim)
        fields.append(ids)
        adv = advancer_ids(ids, sim)

    playoff = run_playoffs(ids, sims_by_stage[-1], M3, seed)
    return dict(sims=sims_by_stage, fields=fields, stages=stages, displays=displays,
                playoff=playoff, gid=gid, universe=universe, paths=paths)


def major_report(pw: Pairwise, res: dict, sims: int, seed: int, thresholds: list[int],
                 out_dir: Path) -> str:
    universe, T = res["universe"], len(res["universe"])
    n = res["sims"][0]["code"].shape[0]
    tot = np.zeros((4, T))
    cnt = np.zeros((4, T))
    for k, (sim, ids) in enumerate(zip(res["sims"], res["fields"]), start=1):
        adv = (sim["code"] <= 2).ravel()
        tot[k] = np.bincount(ids.ravel(), weights=adv, minlength=T)
        cnt[k] = np.bincount(ids.ravel(), minlength=T)
    advance = tot / np.maximum(cnt, 1)

    seeds, stage = res["playoff"]["seeds"], res["playoff"]["stage"]
    ft, fv = seeds.ravel(), stage.ravel()
    semi = np.bincount(ft[fv >= 1], minlength=T) / n
    final = np.bincount(ft[fv >= 2], minlength=T) / n
    champion = np.bincount(ft[fv >= 3], minlength=T) / n

    entry: dict[str, int] = {}
    for k, st in enumerate(res["stages"], start=1):
        for t in st:
            entry.setdefault(t, k)
    pos = {t: i for i, t in enumerate(universe)}

    def through(t: str, upto: int) -> float:
        i = pos[t]
        return float(np.prod([advance[k][i] for k in range(entry[t], upto + 1)]))

    L = ["# Major 全程预测（三阶段瑞士轮 + 季后赛）", "",
         f"- {EVENT_NOTE}",
         "- 种子表：" + "；".join(f"Stage {k} `{p.name}`（{len(d)} 行）"
                                for k, (p, d) in enumerate(zip(res["paths"], res["displays"]), 1)),
         f"- 每阶段 {sims:,} 次蒙特卡洛；随机种子 {seed}；引擎 {pw.engine}；护栏 "
         f"[{pw.rail:.2f}, {1 - pw.rail:.2f}]；建模截止 {pw.model['trained_through']}",
         "- Stage 2/3 的 9-16 号种子由上一阶段模拟结果按最终种子序自动补齐"
         "（种子表写满 16 行则用你写的）。季后赛按规则书全部 BO3。", "",
         "## 各队走到哪一步", "",
         "| 队伍 | 入口 | 进 Stage 2 | 进 Stage 3 | 进季后赛 | 四强 | 决赛 | 夺冠 |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for t in sorted(universe, key=lambda x: -champion[pos[x]]):
        i = pos[t]
        L.append(f"| {pw.names[t]} | Stage {entry[t]} | "
                 f"{1.0 if entry[t] >= 2 else advance[1][i]:.1%} | "
                 f"{1.0 if entry[t] >= 3 else through(t, 2):.1%} | {through(t, 3):.1%} | "
                 f"{semi[i]:.1%} | {final[i]:.1%} | **{champion[i]:.1%}** |")
    L += ["", "「进 Stage 2/3」对直邀队恒为 100%（它们本来就从那一阶段开始）；"
          "「进季后赛」= 从入口一路晋级的连乘。", "",
          "## Stage 2 / Stage 3 的作业怎么出", "",
          "这两个阶段有 8 个名额由上一阶段产生，现在还不知道是谁，所以整届模式只给 Stage 1 的作业；"
          "阶段 2/3 的「各队走到哪一步」已经按全部可能的晋级组合平均好了。",
          "等上一阶段打完，把实际 16 支队伍按种子写进 `stage2.txt` / `stage3.txt`（9-16 号种子 = "
          "上一阶段的最终种子序），再跑：", "",
          "```powershell",
          "python scripts/major_sim.py --stage data/major/singapore2026/stage2.txt",
          "python scripts/major_sim.py --stage data/major/singapore2026/stage3.txt"
          + (" --all-bo3" if res["sims"][2].get("all_bo3", False) else ""),
          "```", "",
          "## 说明", "",
          "- BO3 用的是同一套逐图 p 走 BO3 的 veto 协议与系列赛组合，不是把 BO1 胜率平方。",
          "- 规则书规定季后赛全部 BO3；PGL 新加坡的季军赛与总决赛另按赛事规程，"
          "若总决赛为 BO5，夺冠概率会向强队再偏一点。",
          "- 阶段间在模拟里是独立的（没有把上一阶段的状态带进下一阶段），"
          "这与「每阶段重新按种子表配对阵」的官方做法一致。", ""]
    L += ["", "Stage 3 瑞士轮赛制：" + (
        "全 BO3（--all-bo3）" if res["sims"][2].get("all_bo3", False)
        else "晋级战与淘汰战 BO3，其余 BO1"), ""]
    md = "\n".join(L) + "\n"
    payload = dict(sims=sims, seed=seed, engine=pw.engine, rail=pw.rail,
                   stage3_all_bo3=res["sims"][2].get("all_bo3", False),
                   paths=[p.as_posix() for p in res["paths"]],
                   reach={pw.names[t]: dict(
                       entry=entry[t],
                       to_stage2=1.0 if entry[t] >= 2 else float(advance[1][pos[t]]),
                       to_stage3=1.0 if entry[t] >= 3 else through(t, 2),
                       to_playoffs=through(t, 3), semi=float(semi[pos[t]]),
                       final=float(final[pos[t]]), champion=float(champion[pos[t]]))
                       for t in universe})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "prediction.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return md


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Major 瑞士轮模拟 + 游戏内 Pick'Em 最优作业",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python scripts/major_sim.py --init\n"
               "  python scripts/major_sim.py --stage data/major/singapore2026/stage1.txt\n"
               "  python scripts/major_sim.py --stage stage1.txt --sims 400000 --threshold 6,7\n")
    ap.add_argument("--stage", type=Path, help="种子表：每行一队，从高到低，共 16 行")
    ap.add_argument("--major", nargs=3, type=Path, metavar=("S1", "S2", "S3"),
                    help="整届：三张种子表；Stage 2/3 可只写 8 支直邀，9-16 号自动补齐")
    ap.add_argument("--init", action="store_true", help="写出新加坡 2026 的三张种子表模板")
    ap.add_argument("--sims", type=int, default=120000, help="蒙特卡洛次数，默认 120000")
    ap.add_argument("--all-bo3", action="store_true",
                    help="全 BO3：--stage 时作用于当前阶段；--major 时仅作用于 Stage 3")
    ap.add_argument("--threshold", default="6", help="通过线（答对几个算过），默认 6，即十中六")
    ap.add_argument("--as-of", default="2026-11-24", help="预测日期，默认 2026-11-24（Major 开赛日）")
    ap.add_argument("--engine", choices=["tier", "points"], default="tier")
    ap.add_argument("--rail", type=float, default=RAIL, help=f"对称概率护栏，默认 {RAIL}")
    ap.add_argument("--seed", type=int, default=20261124, help="蒙特卡洛随机种子")
    ap.add_argument("--out", type=Path, default=None, help="输出目录，默认 reports/major/<种子表名>")
    args = ap.parse_args()

    if args.init:
        for p in write_templates():
            print(f"已写出 {p.relative_to(ROOT)}")
        return
    if not args.stage and not args.major:
        ap.error("需要 --stage 单阶段种子表，或 --major 三张种子表，或 --init 生成模板")

    thresholds = [int(x) for x in args.threshold.replace("，", ",").split(",") if x.strip()]
    try:
        pw = Pairwise(args.as_of, engine=args.engine, rail=args.rail)
        if args.major:
            out_dir = args.out or (ROOT / "reports/major/singapore2026")
            res = run_major(pw, list(args.major), args.sims, args.seed, thresholds, out_dir,
                            stage3_all_bo3=args.all_bo3)
            for k, sim in enumerate(res["sims"], start=1):
                if len(res["stages"][k - 1]) < N_TEAMS:
                    print(f"[Stage {k}] 种子表只写了 {len(res['stages'][k - 1])} 支直邀，"
                          f"9-16 号种子未定，本次不给该阶段作业", flush=True)
                    continue
                md, payload, structure = _stage_payload(
                    pw, res["stages"][k - 1], sim, args.sims, args.seed + k, thresholds,
                    f"Stage {k}（整届模式）")
                _write_outputs(out_dir / f"stage{k}", md, payload)
                print(f"[Stage {k}] 结构分布 {structure}；"
                      f"≥{thresholds[0]} 通过率 "
                      f"{payload['solutions'][str(thresholds[0])]['probability']:.1%}", flush=True)
            major_report(pw, res, args.sims, args.seed, thresholds, out_dir)
            print(f"输出：{out_dir.relative_to(ROOT)}")
            return
        out_dir = args.out or (ROOT / "reports/major" / args.stage.stem)
        payload = run_stage(pw, read_seeds(args.stage), args.sims, args.seed, thresholds,
                            out_dir, f"种子表：{args.stage.as_posix()}", all_bo3=args.all_bo3)
    except (ValueError, AssertionError) as exc:
        ap.error(str(exc))
    print(json.dumps({f"≥{k}": payload["solutions"][str(k)]["probability"] for k in thresholds},
                     ensure_ascii=False))
    print(f"输出：{out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir}")


if __name__ == "__main__":
    main()
