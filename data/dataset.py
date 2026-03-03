import kagglehub
import numpy as np
from collections import Counter


def load_data():
    path = kagglehub.dataset_download("yorkyong/text8-zip")
    with open(path + "/text8") as f:
        text = f.read()
    return text.split()


def make_vocabulary(words, args):
    freq = Counter(words)
    vocab_words = [w for w, c in freq.most_common(args.max_vocab) if c >= args.min_count]
    word_to_idx = {w: i for i, w in enumerate(vocab_words)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    data = np.array([word_to_idx[w] for w in words if w in word_to_idx], dtype=np.int32)
    print(f"Total tokens (after filtering): {len(data):,}")
    print(f"Vocab size: {len(vocab_words):,}")
    return data, idx_to_word, word_to_idx, vocab_words, freq


def compute_keep_prob(freq, idx_to_word, vocab_size, args):
    total_tokens = sum(freq.values())
    word_counts = np.array([freq[idx_to_word[i]] for i in range(vocab_size)], dtype=np.float32)
    word_freqs = word_counts / total_tokens
    keep_prob = np.minimum(
        np.sqrt(args.threshold / word_freqs) + (args.threshold / word_freqs), 1.0
    )
    return keep_prob, word_freqs


def apply_subsampling(data, keep_prob):
    mask = keep_prob[data] > np.random.rand(len(data))
    return data[mask]


def build_noise_dist(word_freqs):
    noise_dist = word_freqs ** 0.75
    noise_dist /= noise_dist.sum()
    return noise_dist


def sample_negatives(center, context, k, noise_dist):
    vocab_size = len(noise_dist)
    negs = np.random.choice(vocab_size, size=k * 3, p=noise_dist)
    negs = negs[(negs != center) & (negs != context)]
    return negs[:k]


def generate_pairs(data, max_window, center_start=None, center_end=None):
    n = len(data)
    W = max_window
    if center_start is None:
        center_start = 0
    if center_end is None:
        center_end = n

    all_centers, all_contexts = [], []

    for d in range(-W, W + 1):
        if d == 0:
            continue

        lo = center_start
        hi = center_end
        if d > 0:
            hi = min(hi, n - d)
        else:
            lo = max(lo, -d)

        if lo >= hi:
            continue

        # P(include offset d) = (W - |d| + 1) / W  →  equivalent to P(window >= |d|)
        p    = (W - abs(d) + 1) / W
        keep = np.random.rand(hi - lo) < p
        pos  = np.where(keep)[0] + lo

        if len(pos) == 0:
            continue

        all_centers.append(data[pos])
        all_contexts.append(data[pos + d])

    if not all_centers:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    centers  = np.concatenate(all_centers).astype(np.int32)
    contexts = np.concatenate(all_contexts).astype(np.int32)

    perm = np.random.permutation(len(centers))
    return centers[perm], contexts[perm]
