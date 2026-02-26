import kagglehub
import numpy as np
from collections import Counter


def load_data():
    path = kagglehub.dataset_download("yorkyong/text8-zip")
    filepath = path + "/text8"

    with open(filepath, "r") as f:
        text = f.read()

    words = text.split()
    return words


def make_vocabulary(words, args):
    freq = Counter(words)

    # Sort by frequency descending, apply min-count filter, cap size
    vocab_words = [w for w, c in freq.most_common(args.max_vocab) if c >= args.min_count]
    word_to_idx = {w: i for i, w in enumerate(vocab_words)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    # Map corpus to integer IDs, drop unknown words
    data = np.array([word_to_idx[w] for w in words if w in word_to_idx], dtype=np.int32)

    print(f"Total tokens (after filtering): {len(data):,}")
    print(f"Vocab size: {len(vocab_words):,}")

    return data, idx_to_word, word_to_idx, vocab_words, freq


def subsampling(data, freq, idx_to_word, vocab_size, args):
    """
    Mikolov et al.: discard word w with prob = 1 - sqrt(t / f(w)).
    This speeds up training and improves embeddings for rare words.
    Returns the filtered data array and the word frequency array.
    """
    word_counts = np.array([freq[idx_to_word[i]] for i in range(vocab_size)], dtype=np.float32)
    word_freqs = word_counts / word_counts.sum()
    keep_prob = np.minimum(np.sqrt(args.threshold / word_freqs), 1.0)

    # Apply subsampling to the corpus
    mask = keep_prob[data] > np.random.rand(len(data))
    data = data[mask]

    print(f"Total tokens after subsampling: {len(data):,}")

    return data, word_freqs


def build_noise_dist(word_freqs):
    """
    Compute the unigram^(3/4) distribution for negative sampling.
    Precomputing this once avoids redundant work on every training step.
    """
    noise_dist = word_freqs ** 0.75
    noise_dist /= noise_dist.sum()
    return noise_dist


def sample_negatives(center, context, k, noise_dist):
    """Draw k negative samples, excluding center and context."""
    vocab_size = len(noise_dist)
    negs = np.random.choice(vocab_size, size=k * 3, p=noise_dist)
    negs = negs[(negs != center) & (negs != context)]
    return negs[:k]
