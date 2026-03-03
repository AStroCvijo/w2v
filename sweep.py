"""
Hyperparameter sweep — trains each config and evaluates on both benchmarks.
Results are printed as a markdown table at the end.

Usage:
    python sweep.py
    python sweep.py --analogy questions-words.txt --similarity combined.tab
"""
import argparse
import os
import re
import subprocess
import sys


CONFIGS = [
    {"epochs": 5,  "embed_dim": 200, "neg_k":  5, "window": 5},
    {"epochs": 5,  "embed_dim": 200, "neg_k": 10, "window": 5},
    {"epochs": 5,  "embed_dim": 300, "neg_k":  5, "window": 5},
    {"epochs": 10, "embed_dim": 200, "neg_k":  5, "window": 5},
    {"epochs": 10, "embed_dim": 200, "neg_k": 10, "window": 5},
    {"epochs": 10, "embed_dim": 300, "neg_k": 10, "window": 5},
]


def run(cmd):
    """Stream command output to stdout and return the full output string."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    return "".join(lines)


def parse_analogy(output):
    m = re.search(r"OVERALL\s+\d+\s+\d+\s+([\d.]+)%", output)
    return float(m.group(1)) if m else None


def parse_similarity(output):
    m = re.search(r"Spearman ρ\s*:\s*([\d.]+)", output)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analogy",    default="questions-words.txt")
    parser.add_argument("--similarity", default="combined.tab")
    args = parser.parse_args()

    results = []

    for cfg in CONFIGS:
        tag     = f"e{cfg['epochs']}_d{cfg['embed_dim']}_k{cfg['neg_k']}_w{cfg['window']}"
        emb_dir = f"embeddings/sweep/{tag}"
        w_in    = f"{emb_dir}/W_in.npy"
        w_out   = f"{emb_dir}/W_out.npy"
        vocab   = f"{emb_dir}/vocab.txt"

        print(f"\n{'='*60}")
        print(f"  epochs={cfg['epochs']}  embed-dim={cfg['embed_dim']}  neg-k={cfg['neg_k']}  window={cfg['window']}")
        print(f"{'='*60}\n")

        if os.path.exists(w_in):
            print(f"  [skipping training — {w_in} already exists]\n")
        else:
            run([
                sys.executable, "main.py",
                "--epochs",              str(cfg["epochs"]),
                "--embed-dim",           str(cfg["embed_dim"]),
                "--neg-k",               str(cfg["neg_k"]),
                "--window",              str(cfg["window"]),
                "--w-in-path",           w_in,
                "--w-out-path",          w_out,
                "--vocab-path",          vocab,
                "--checkpoint-interval", "0",
            ])

        eval_cmd = [
            sys.executable, "eval.py",
            "--w-in-path",  w_in,
            "--vocab-path", vocab,
        ]
        if os.path.exists(args.analogy):
            eval_cmd += ["--benchmark-analogy", args.analogy]
        if os.path.exists(args.similarity):
            eval_cmd += ["--benchmark-similarity", args.similarity]

        out = run(eval_cmd)
        results.append((cfg, parse_analogy(out), parse_similarity(out)))

    print(f"\n\n{'='*60}")
    print("SWEEP RESULTS")
    print(f"{'='*60}\n")
    print("| epochs | embed-dim | neg-k | window | Analogy % | WordSim ρ |")
    print("|--------|-----------|-------|--------|-----------|-----------|")
    for cfg, analogy, sim in results:
        a = f"{analogy:.1f}%" if analogy is not None else "—"
        s = f"{sim:.4f}"      if sim     is not None else "—"
        print(f"| {cfg['epochs']:>6} | {cfg['embed_dim']:>9} | {cfg['neg_k']:>5} | {cfg['window']:>6} | {a:>9} | {s:>9} |")
    print()


if __name__ == "__main__":
    main()
