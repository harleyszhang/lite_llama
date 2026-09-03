"""Per-step divergence anatomy for the V4 parity JSONs.

Reads the lite and HF arm JSONs produced by ``accuracy_v4_parity.py`` and, for
every step where the greedy tokens disagree, prints the decision margin of
both arms — the top1-top2 logprob gap and where the other arm's token ranks
in this arm's top-5. A divergence is numerically benign when the winning
margin is tiny and the loser sits at rank 2-3; a structural bug shows up as
divergent picks from clearly-separated distributions.

Usage:
    python benchmarks/analysis_v4_divergence.py LITE.json HF.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    return {p["seq_len"]: p for p in data["prompts"]}


def margin(top5: list[list]) -> float:
    """top1 - top2 logprob gap of one step's top-5."""
    return top5[0][1] - top5[1][1]


def rank_of(top5: list[list], tid: int) -> int | None:
    for i, (i_, _) in enumerate(top5):
        if i_ == tid:
            return i + 1
    return None


def main() -> None:
    lite = load(sys.argv[1])
    hf = load(sys.argv[2])

    for seq_len in sorted(lite):
        lp, hp = lite[seq_len], hf[seq_len]
        divs = [
            i for i, (a, b) in enumerate(zip(lp["greedy_tokens"], hp["greedy_tokens"])) if a != b
        ]
        print(f"=== seq {seq_len}: {len(lp['greedy_tokens']) - len(divs)}/{len(lp['greedy_tokens'])} greedy match, divergent steps {divs}")
        for i in divs:
            lt, ht = lp["steps"][i]["top5"], hp["steps"][i]["top5"]
            l_tok, h_tok = lp["greedy_tokens"][i], hp["greedy_tokens"][i]
            # where each arm's pick stands in the other arm's top-5
            h_rank_in_lite = rank_of(lt, h_tok)
            l_rank_in_hf = rank_of(ht, l_tok)
            lp_of_h_pick = next((v for t, v in lt if t == h_tok), None)
            hp_of_l_pick = next((v for t, v in ht if t == l_tok), None)
            print(
                f"  step {i:>2}: lite={l_tok} hf={h_tok} | "
                f"lite margin {margin(lt):+.4f} hf margin {margin(ht):+.4f} | "
                f"hf-pick rank-in-lite {h_rank_in_lite} (lp {lp_of_h_pick and round(lp_of_h_pick, 3)}) "
                f"lite-pick rank-in-hf {l_rank_in_hf} (lp {hp_of_l_pick and round(hp_of_l_pick, 3)})"
            )
            if h_rank_in_lite is None or l_rank_in_hf is None:
                # loser fell outside the winner's top-5: dump both top-5s
                print(f"    lite top5: {[(t, round(v, 3)) for t, v in lt]}")
                print(f"    hf   top5: {[(t, round(v, 3)) for t, v in ht]}")
        print()


if __name__ == "__main__":
    main()
