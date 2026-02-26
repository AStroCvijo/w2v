import numpy as np

from data.dataset import load_data, make_vocabulary, subsampling, build_noise_dist
from model.model import build_model, train, nearest_neighbours
from utils.args import parse_args


if __name__ == "__main__":

    # Parse arguments
    args = parse_args()

    # Load and tokenize
    words = load_data()

    # Build vocabulary
    data, idx_to_word, word_to_idx, vocab_words, freq = make_vocabulary(words, args)

    # Subsample frequent words (returns filtered corpus + word frequencies)
    data, word_freqs = subsampling(data, freq, idx_to_word, len(vocab_words), args)

    # Precompute unigram^(3/4) distribution for negative sampling
    noise_dist = build_noise_dist(word_freqs)

    # Initialize embedding matrices
    W_in, W_out = build_model(len(vocab_words), args.embed_dim)

    # Train
    train(data, noise_dist, W_in, W_out, args)

    # Evaluate nearest neighbours
    nearest_neighbours("king",    word_to_idx, idx_to_word, W_in)
    nearest_neighbours("paris",   word_to_idx, idx_to_word, W_in)
    nearest_neighbours("science", word_to_idx, idx_to_word, W_in)

    # Save embeddings and vocabulary
    np.save("embeddings_W_in.npy",  W_in)
    np.save("embeddings_W_out.npy", W_out)
    with open("vocab.txt", "w") as f:
        for i in range(len(vocab_words)):
            f.write(idx_to_word[i] + "\n")
    print("\nEmbeddings saved.")
