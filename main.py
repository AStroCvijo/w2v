import numpy as np
import random

from data.dataset import load_data, make_vocabulary, compute_keep_prob, build_noise_dist
from model.model import build_model, train, nearest_neighbours
from utils.args import parse_args
from utils.checkpoint import save_final


if __name__ == "__main__":

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    words = load_data()
    data, idx_to_word, word_to_idx, vocab_words, freq = make_vocabulary(words, args)

    keep_prob, word_freqs = compute_keep_prob(freq, idx_to_word, len(vocab_words), args)
    noise_dist = build_noise_dist(word_freqs)

    W_in, W_out = build_model(len(vocab_words), args.embed_dim, args.seed)

    train(data, keep_prob, noise_dist, W_in, W_out, args)

    save_final(W_in, W_out, idx_to_word, vocab_words, args)
