"""Build submission.tar.gz for a given agent directory (expects main.py, deck.csv, cg/,
plus any other top-level .py/.json files main.py imports/loads -- e.g. bc_reranker_v2's
features.py, heuristic_scored.py, model.json)."""
import os
import sys
import tarfile

def build(agent_dir: str, out_path: str | None = None) -> str:
    agent_dir = os.path.abspath(agent_dir)
    main_py = os.path.join(agent_dir, "main.py")
    deck_csv = os.path.join(agent_dir, "deck.csv")
    cg_dir = os.path.join(agent_dir, "cg")
    for p in (main_py, deck_csv, cg_dir):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    out_path = out_path or os.path.join(agent_dir, "submission.tar.gz")
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(main_py, arcname="main.py")
        tar.add(cg_dir, arcname="cg")
        tar.add(deck_csv, arcname="deck.csv")
        for name in sorted(os.listdir(agent_dir)):
            full = os.path.join(agent_dir, name)
            if name in ("main.py", "deck.csv", "cg", "submission.tar.gz"):
                continue
            if name.startswith("__") or name.startswith("."):
                continue
            if os.path.isfile(full) and (name.endswith(".py") or name.endswith(".json")):
                tar.add(full, arcname=name)
                print(f"  + extra file: {name}")
    return out_path

if __name__ == "__main__":
    agent_dir = sys.argv[1]
    out = build(agent_dir)
    size_mb = os.path.getsize(out) / 1e6
    print(f"Built {out} ({size_mb:.2f} MB)")
