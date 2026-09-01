"""
Memory-safe two-pass subsampler for the cross-play row shards (21M rows / ~19GB
across 18 shard files -- far too large to load into a Python list at once, see
AGENT_LOG.md's cross-play generation entry). For each shard file independently:
  pass 1: fast-parse just the leading `"group": <int>` token of every line
          (rows are written with "group" as the first JSON key) to enumerate
          that shard's distinct decision-group ids without building any row
          objects.
  pass 2: randomly pick this shard's share of the global group-count target,
          re-stream the shard, keep (fully json-parse) only rows belonging to
          a kept group, renumber groups into a globally-unique range, write to
          the combined output file.

Usage:
  python3 research/subsample_crossplay.py <shard_glob_dir> <out_jsonl> <total_group_target> [seed]
"""
import glob
import json
import os
import random
import sys
import time


def fast_group_id(line):
    # rows are written as {"group": <int>, "label": ... -- extract the int
    # without a full json.loads call.
    start = line.index(b'"group": ') + len(b'"group": ')
    end = line.index(b',', start)
    return int(line[start:end])


def main():
    shard_dir = sys.argv[1]
    out_path = sys.argv[2]
    total_target = int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    shard_files = sorted(glob.glob(os.path.join(shard_dir, "rows_shard_*.jsonl")))
    print(f"Found {len(shard_files)} shard files")
    per_shard_target = max(1, total_target // len(shard_files))
    print(f"Target ~{per_shard_target} groups/shard (~{per_shard_target*len(shard_files)} total)")

    rng = random.Random(seed)
    next_group_id = 1
    total_rows_written = 0
    total_groups_seen = 0
    total_groups_kept = 0
    t0 = time.time()

    with open(out_path, "w") as out_f:
        for shard_fp in shard_files:
            # pass 1: enumerate distinct group ids in this shard (byte-mode, fast)
            groups_here = set()
            with open(shard_fp, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        gid = fast_group_id(line)
                    except Exception:
                        continue
                    groups_here.add(gid)
            groups_list = sorted(groups_here)
            rng.shuffle(groups_list)
            keep_set = set(groups_list[:per_shard_target])
            total_groups_seen += len(groups_here)
            total_groups_kept += len(keep_set)

            # pass 2: re-stream, keep only rows in a kept group, renumber, write
            local_remap = {}
            with open(shard_fp, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        gid = fast_group_id(line)
                    except Exception:
                        continue
                    if gid not in keep_set:
                        continue
                    row = json.loads(line)
                    if row["group"] not in local_remap:
                        local_remap[row["group"]] = next_group_id
                        next_group_id += 1
                    row["group"] = local_remap[row["group"]]
                    out_f.write(json.dumps(row) + "\n")
                    total_rows_written += 1
            print(f"  {os.path.basename(shard_fp)}: {len(groups_here)} groups seen, "
                  f"{len(keep_set)} kept, rows written so far={total_rows_written} "
                  f"({time.time()-t0:.1f}s elapsed)", flush=True)

    print(f"\nDONE: {total_groups_seen} groups seen total, {total_groups_kept} kept, "
          f"{total_rows_written} rows written to {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
