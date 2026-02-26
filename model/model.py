import numpy as np
import time

from data.dataset import sample_negatives


def build_model(vocab_size, embed_dim):
    """Initialize center-word (W_in) and context-word (W_out) embedding matrices."""
    rng = np.random.default_rng(42)
    W_in  = rng.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim)).astype(np.float32)
    W_out = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    return W_in, W_out


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def train_step(center, context, neg_samples, lr, W_in, W_out):
    """
    Single skip-gram negative-sampling update.

    Loss = -log σ(v_c · u_o)  -  Σ_k log σ(-v_c · u_k)

    Gradients:
      ∂L/∂v_c = (σ(v_c·u_o) - 1)·u_o  +  Σ_k σ(v_c·u_k)·u_k
      ∂L/∂u_o = (σ(v_c·u_o) - 1)·v_c
      ∂L/∂u_k = σ(v_c·u_k)·v_c
    """
    v_c = W_in[center]       # (D,)
    u_o = W_out[context]     # (D,)
    u_n = W_out[neg_samples] # (K, D)

    # Forward
    pos_score = np.dot(v_c, u_o)  # scalar
    neg_scores = u_n @ v_c        # (K,)

    pos_sig = sigmoid(pos_score)  # scalar
    neg_sig = sigmoid(neg_scores) # (K,)

    # Loss (for logging only)
    loss = -np.log(pos_sig + 1e-9) - np.sum(np.log(1 - neg_sig + 1e-9))

    # Gradients
    grad_vc = (pos_sig - 1) * u_o + neg_sig @ u_n  # (D,)
    grad_uo = (pos_sig - 1) * v_c                  # (D,)
    grad_un = np.outer(neg_sig, v_c)               # (K, D)

    # In-place parameter updates
    W_in[center]       -= lr * grad_vc
    W_out[context]     -= lr * grad_uo
    W_out[neg_samples] -= lr * grad_un

    return loss


def train(data, noise_dist, W_in, W_out, args):
    """Main training loop: skip-gram with negative sampling."""
    n_tokens    = len(data)
    total_steps = n_tokens * args.epochs
    step        = 0
    lr          = args.lr_init  # initialised here so it's always defined for logging

    for epoch in range(args.epochs):
        total_loss   = 0.0
        total_steps_epoch = 0
        recent_loss  = 0.0
        recent_steps = 0
        t0 = time.time()

        for i in range(n_tokens):
            center = int(data[i])

            # Linear LR decay based on tokens processed, not pairs.
            # step (pair count) grows ~window× faster than i, so using step
            # would exhaust the LR budget after only a fraction of the corpus.
            lr = max(args.lr_init * (1 - (epoch * n_tokens + i) / total_steps), 0.0001)

            # Dynamic window: sample actual window size uniformly in [1, window]
            w     = np.random.randint(1, args.window + 1)
            start = max(0, i - w)
            end   = min(n_tokens, i + w + 1)

            for j in range(start, end):
                if j == i:
                    continue

                context     = int(data[j])
                neg_samples = sample_negatives(center, context, args.neg_k, noise_dist)
                if len(neg_samples) < args.neg_k:
                    continue  # skip if we couldn't fill negatives

                loss              = train_step(center, context, neg_samples, lr, W_in, W_out)
                total_loss       += loss
                total_steps_epoch += 1
                recent_loss      += loss
                recent_steps     += 1
                step             += 1

            if (i + 1) % args.log_interval == 0:
                elapsed      = time.time() - t0
                epoch_avg    = total_loss / max(total_steps_epoch, 1)
                recent_avg   = recent_loss / max(recent_steps, 1)
                print(f"Epoch {epoch+1} | Token {i+1:,}/{n_tokens:,} | "
                      f"Loss (epoch avg): {epoch_avg:.4f} | Loss (recent): {recent_avg:.4f} | "
                      f"LR: {lr:.5f} | Time: {elapsed:.1f}s")
                recent_loss  = 0.0
                recent_steps = 0

    print("Training complete.")


def cosine_similarity(v, M):
    """Cosine similarity between vector v and every row of matrix M."""
    norms = np.linalg.norm(M, axis=1) + 1e-9
    return (M @ v) / (norms * (np.linalg.norm(v) + 1e-9))


def nearest_neighbours(word, word_to_idx, idx_to_word, W_in, top_n=10):
    """Print the top_n nearest neighbours to a word by cosine similarity."""
    if word not in word_to_idx:
        print(f"'{word}' not in vocab.")
        return
    idx  = word_to_idx[word]
    vec  = W_in[idx]
    sims = cosine_similarity(vec, W_in)
    sims[idx] = -1  # exclude the word itself
    top_idx = np.argsort(sims)[::-1][:top_n]
    print(f"\nNearest neighbours to '{word}':")
    for i in top_idx:
        print(f"  {idx_to_word[i]:<20} {sims[i]:.4f}")
