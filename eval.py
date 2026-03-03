import argparse
import re
import numpy as np

from model.model import cosine_similarity


# Loading

def load_embeddings(w_in_path, vocab_path):
    W_in = np.load(w_in_path)
    idx_to_word = {}
    word_to_idx = {}
    with open(vocab_path) as f:
        for i, line in enumerate(f):
            word = line.rstrip("\n")
            idx_to_word[i] = word
            word_to_idx[word] = i
    return W_in, word_to_idx, idx_to_word


# REPL

def eval_expr(expr, word_to_idx, W_in):
    # Parse a word arithmetic expression into a result vector.
    tokens = re.findall(r'[+-]|\w+', expr)
    vec  = None
    sign = 1
    used = []
    for tok in tokens:
        if tok == '+':
            sign = 1
        elif tok == '-':
            sign = -1
        else:
            if tok not in word_to_idx:
                raise KeyError(tok)
            idx = word_to_idx[tok]
            used.append(idx)
            if vec is None:
                vec = sign * W_in[idx].astype(np.float64)
            else:
                vec += sign * W_in[idx]
            sign = 1
    if vec is None:
        raise ValueError("empty expression")
    return vec, used


def nearest(vec, W_in, idx_to_word, top_n, exclude=()):
    # Return top_n (word, cosine_similarity) pairs, skipping excluded indices
    sims = cosine_similarity(vec, W_in)
    for i in exclude:
        sims[i] = -2.0
    top_idx = np.argsort(sims)[::-1][:top_n]
    return [(idx_to_word[i], float(sims[i])) for i in top_idx]


def repl(W_in, word_to_idx, idx_to_word, top_n):
    print("Word2Vec eval — enter an expression or 'quit' to exit.")
    print("  nn <word>              nearest neighbours")
    print("  king - man + woman     vector arithmetic / analogy")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break

        query = line[3:].strip() if line.startswith("nn ") else line

        try:
            vec, used = eval_expr(query, word_to_idx, W_in)
        except KeyError as e:
            print(f"  '{e.args[0]}' not in vocabulary\n")
            continue
        except ValueError:
            print("  Empty expression\n")
            continue

        results = nearest(vec, W_in, idx_to_word, top_n, exclude=used)
        label = query if len(query) <= 40 else query[:37] + "..."
        print(f"\n  Nearest to [{label}]:")
        for word, sim in results:
            print(f"    {word:<22} {sim:.4f}")
        print()


# Benchmarks

# Sections from the Google analogy file that are semantic
_SEMANTIC = {
    "capital-common-countries", "capital-world",
    "currency", "city-in-state", "family",
}


def _spearman(x, y):
    # Spearman rank correlation computed as Pearson on ranks
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(np.dot(rx, ry) / denom) if denom > 0 else 0.0


def run_analogy_benchmark(W_in, word_to_idx, idx_to_word, path, batch_size=1024):
    # Evaluate on the Google analogy dataset (Mikolov et al., 2013).
    print(f"Analogy benchmark  ({path})\n")

    # Pre-normalise once — dot product on unit vectors == cosine similarity
    norms  = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-9
    W_norm = (W_in / norms).astype(np.float32)

    # Parse file
    section  = "misc"
    sec_data = {}   # section → {correct, total, skipped}
    sec_order = []
    questions = []  # (section, a_i, b_i, c_i, d_i)

    def _sec(name):
        if name not in sec_data:
            sec_data[name] = {"correct": 0, "total": 0, "skipped": 0}
            sec_order.append(name)
        return sec_data[name]

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip().lower()
            if not line:
                continue
            if line.startswith(":"):
                section = line[1:].strip()
                _sec(section)
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            a, b, c, d = parts
            s = _sec(section)
            if any(w not in word_to_idx for w in (a, b, c, d)):
                s["skipped"] += 1
                continue
            s["total"] += 1
            questions.append((section,
                               word_to_idx[a], word_to_idx[b],
                               word_to_idx[c], word_to_idx[d]))

    if not questions:
        print("  No answerable questions found (all words OOV?).")
        return

    # Batched evaluation
    n = len(questions)
    for start in range(0, n, batch_size):
        batch = questions[start:start + batch_size]

        q_vecs = np.stack([
            W_norm[b_i] - W_norm[a_i] + W_norm[c_i]
            for _, a_i, b_i, c_i, _ in batch
        ])  # (B, D)

        q_norms       = np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-9
        q_vecs        = (q_vecs / q_norms).astype(np.float32)
        sims          = q_vecs @ W_norm.T  # (B, V) - cosine similarity

        for i, (sec, a_i, b_i, c_i, d_i) in enumerate(batch):
            sims[i, a_i] = -2.0
            sims[i, b_i] = -2.0
            sims[i, c_i] = -2.0
            if int(np.argmax(sims[i])) == d_i:
                sec_data[sec]["correct"] += 1

    # Print results
    W = 34  # label column width

    def _row(label, c, t, sk, indent=0):
        acc = f"{100*c/t:5.1f}%" if t > 0 else "    —  "
        pad = "  " * indent
        print(f"  {pad}{label:<{W - 2*indent}} {c:>7}  {t:>7}  {acc}  {sk:>8}")

    header = f"  {'Section':<{W}} {'Correct':>7}  {'Total':>7}  {'Acc':>7}  {'Skipped':>8}"
    rule   = "  " + "─" * (len(header) - 2)
    print(header)
    print(rule)

    sem_c = sem_t = sem_s = 0
    syn_c = syn_t = syn_s = 0

    print("  Semantic")
    for sec in sec_order:
        if sec not in _SEMANTIC:
            continue
        d = sec_data[sec]
        _row(sec, d["correct"], d["total"], d["skipped"], indent=1)
        sem_c += d["correct"]; sem_t += d["total"]; sem_s += d["skipped"]
    _row("SEMANTIC TOTAL", sem_c, sem_t, sem_s, indent=1)

    print("  Syntactic")
    for sec in sec_order:
        if sec in _SEMANTIC or sec == "misc":
            continue
        d = sec_data[sec]
        _row(sec, d["correct"], d["total"], d["skipped"], indent=1)
        syn_c += d["correct"]; syn_t += d["total"]; syn_s += d["skipped"]
    _row("SYNTACTIC TOTAL", syn_c, syn_t, syn_s, indent=1)

    print(rule)
    tot_c = sem_c + syn_c
    tot_t = sem_t + syn_t
    tot_s = sem_s + syn_s
    _row("OVERALL", tot_c, tot_t, tot_s)
    print()


def run_similarity_benchmark(W_in, word_to_idx, path):
    # Evaluate on a word-similarity dataset (e.g. WordSim-353)
    print(f"Similarity benchmark  ({path})\n")

    model_sims  = []
    human_sims  = []
    skipped_oov = 0
    skipped_fmt = 0

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Support tab- and comma-separated files
            parts = line.split("\t") if "\t" in line else line.split(",")
            if len(parts) < 3:
                skipped_fmt += 1
                continue
            w1, w2 = parts[0].strip().lower(), parts[1].strip().lower()
            try:
                score = float(parts[2].strip())
            except ValueError:
                skipped_fmt += 1  # header row or malformed
                continue
            if w1 not in word_to_idx or w2 not in word_to_idx:
                skipped_oov += 1
                continue
            v1 = W_in[word_to_idx[w1]]
            v2 = W_in[word_to_idx[w2]]
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            model_sims.append(cos)
            human_sims.append(score)

    n = len(model_sims)
    if n < 2:
        print(f"  Too few pairs to evaluate (answered {n}, skipped OOV={skipped_oov}).")
        return

    rho = _spearman(np.array(human_sims), np.array(model_sims))
    print(f"  Pairs evaluated : {n}")
    print(f"  Skipped (OOV)   : {skipped_oov}")
    print(f"  Spearman ρ      : {rho:.4f}")
    print()


# CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Word2Vec — eval / inspection mode")
    parser.add_argument("--w-in-path",  default="embeddings/W_in.npy", help="Path to W_in embeddings (.npy)")
    parser.add_argument("--vocab-path", default="embeddings/vocab.txt", help="Path to vocabulary file")
    parser.add_argument("--top-n",      type=int, default=10,           help="Nearest neighbours to show in REPL")
    parser.add_argument("--benchmark-analogy",    metavar="FILE",       help="Path to Google analogy file (questions-words.txt)")
    parser.add_argument("--benchmark-similarity", metavar="FILE",       help="Path to word similarity file (e.g. wordsim353.tsv)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Loading {args.w_in_path} ...")
    W_in, word_to_idx, idx_to_word = load_embeddings(args.w_in_path, args.vocab_path)
    print(f"Loaded {W_in.shape[0]:,} words × {W_in.shape[1]} dims\n")

    if args.benchmark_analogy or args.benchmark_similarity:
        if args.benchmark_analogy:
            run_analogy_benchmark(W_in, word_to_idx, idx_to_word, args.benchmark_analogy)
        if args.benchmark_similarity:
            run_similarity_benchmark(W_in, word_to_idx, args.benchmark_similarity)
    else:
        repl(W_in, word_to_idx, idx_to_word, args.top_n)
