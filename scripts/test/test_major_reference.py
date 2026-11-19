"""与用户提供的参考项目逐次对照瑞士轮逻辑（参考目录只读）。

python scripts/test/test_major_reference.py --reference C:/path/to/cs2-major-pickems-simulation

给双方使用相同的单次系列赛抽样器，以隔离随机数消耗方式的区别。
参考项目的正式配对、排序和轮次推进代码保持原样执行。
依赖参考项目所需的 tqdm；常规测试不依赖此文件或参考目录。
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import major_sim as ms  # noqa: E402


def compare(reference: Path, samples: int, seed: int) -> None:
    spec = importlib.util.spec_from_file_location("pickem_reference", reference / "simulate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    teams = [module.Team(id=i, name=f"T{i}", seed=i + 1, rating=()) for i in range(16)]
    rng = np.random.default_rng(seed)
    raw = rng.uniform(.15, .85, (16, 16))
    p1 = raw / (raw + raw.T)
    np.fill_diagonal(p1, .5)
    # 参考项目三图使用同一 p；其系列赛胜率恰为 3p² - 2p³。
    p3 = 3 * p1 ** 2 - 2 * p1 ** 3

    for all_bo3 in (False, True):
        incomplete = []

        class ReferenceSwiss(module.SwissSystem):
            def simulate_match(self, a, b):
                ra, rb = self.records[a], self.records[b]
                bo3 = all_bo3 or ra.wins == 2 or ra.losses == 2
                p = (p3 if bo3 else p1)[a.id, b.id]
                winner, loser = (a, b) if draw() < p else (b, a)
                self.records[winner].wins += 1
                self.records[loser].losses += 1
                ra.teams_faced.add(b)
                rb.teams_faced.add(a)
                for team in (a, b):
                    r = self.records[team]
                    if r.wins == 3 or r.losses == 3:
                        self.remaining.remove(team)

        for s in range(samples):
            trial_seed = seed + s
            draw = random.Random(trial_seed).random
            swiss = ReferenceSwiss({}, {t: module.Record.new() for t in teams}, set(teams))
            swiss.simulate_tournament()
            ids = np.arange(16, dtype=np.int16)[None, :]
            actual = ms.simulate(ids, p1, p3, seed=trial_seed, all_bo3=all_bo3)
            ms.check_structure(actual["code"])
            assert actual["rematches"] == 0
            expected_counts = (0.0, 33.0) if all_bo3 else (20.0, 13.0)
            assert (actual["bo1_matches"], actual["bo3_matches"]) == expected_counts
            if swiss.remaining:
                # 参考实现第 2/3 轮贪心匹配失败时会漏赛，不能将其当作正确答案。
                incomplete.append(trial_seed)
                continue
            expected_code = np.empty((1, 16), np.int8)
            expected_difficulty = np.empty((1, 16), np.int16)
            for t in teams:
                record = swiss.records[t]
                expected_code[0, t.id] = ms.CODE_OF[(record.wins, record.losses)]
                expected_difficulty[0, t.id] = sum(swiss.records[o].diff for o in record.teams_faced)
            np.testing.assert_array_equal(actual["code"], expected_code,
                                          err_msg=f"trial_seed={trial_seed}")
            np.testing.assert_array_equal(actual["difficulty"], expected_difficulty,
                                          err_msg=f"trial_seed={trial_seed}")
        assert len(incomplete) < samples, "参考实现没有完成任何可对照样本"
        print(f"PASS: all_bo3={all_bo3}, {samples - len(incomplete)} 次完整参考样本的战绩及 Buchholz 完全一致")
        print(f"参考实现另有 {len(incomplete)} 次漏赛；本实现全部 {samples} 次结构及场数检查通过。"
              f"前 5 个漏赛种子：{incomplete[:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    compare(args.reference, args.samples, args.seed)
