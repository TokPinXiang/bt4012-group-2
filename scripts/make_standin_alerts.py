"""Write a stand-in prediction file so the agent can be built before the models exist.

    python scripts/make_standin_alerts.py

Output: outputs/predictions/predictions_standin.csv (shared prediction format).
Swap in predictions_<model>.csv from Members 2 and 3 when ready.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.alerts import make_standin_alerts  # noqa: E402
from src.agent.store import find_data_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=find_data_dir(ROOT))
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/predictions/predictions_standin.csv")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--per-typology", type=int, default=3)
    ap.add_argument("--n-unlisted", type=int, default=6)
    ap.add_argument("--n-normal-hard", type=int, default=10)
    ap.add_argument("--n-normal-random", type=int, default=10)
    ap.add_argument("--seed", type=int, default=4012)
    a = ap.parse_args()

    df = make_standin_alerts(
        a.data_dir, a.out, split=a.split, per_typology=a.per_typology, n_unlisted=a.n_unlisted,
        n_normal_hard=a.n_normal_hard, n_normal_random=a.n_normal_random, seed=a.seed,
    )
    print(f"Wrote {len(df)} stand-in alerts to {a.out}")


if __name__ == "__main__":
    main()
