import numpy as np
import time

from data.dataset import generate_pairs, apply_subsampling
from utils.checkpoint import save_checkpoint


def build_model(vocab_size, embed_dim, seed=42):
    rng = np.random.default_rng(seed)
    W_in  = rng.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim)).astype(np.float32)
    W_out = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    return W_in, W_out


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def train_step(center, context, neg_samples, lr, W_in, W_out):
    v_c = W_in[center]       # (D,)
    u_o = W_out[context]     # (D,)
    u_n = W_out[neg_samples] # (K, D)

    pos_score = np.dot(v_c, u_o)
    neg_scores = u_n @ v_c

    pos_sig = sigmoid(pos_score)
    neg_sig = sigmoid(neg_scores)

    loss = -np.log(pos_sig + 1e-9) - np.sum(np.log(1 - neg_sig + 1e-9))

    grad_vc = (pos_sig - 1) * u_o + neg_sig @ u_n  # (D,)
    grad_uo = (pos_sig - 1) * v_c                  # (D,)
    grad_un = np.outer(neg_sig, v_c)               # (K, D)

    W_in[center]       -= lr * grad_vc
    W_out[context]     -= lr * grad_uo
    W_out[neg_samples] -= lr * grad_un

    return loss


def train_step_batch(centers, contexts, neg_samples, lr, W_in, W_out):
    """
    centers     : int32 (B,)
    contexts    : int32 (B,)
    neg_samples : int32 (B, K)
    """
    D = W_in.shape[1]

    v_c = W_in[centers]       # (B, D)
    u_o = W_out[contexts]     # (B, D)
    u_n = W_out[neg_samples]  # (B, K, D)

    pos_score  = np.einsum('bd,bd->b',   v_c, u_o)   # (B,)
    neg_scores = np.einsum('bd,bkd->bk', v_c, u_n)   # (B, K)

    pos_sig = sigmoid(pos_score)   # (B,)
    neg_sig = sigmoid(neg_scores)  # (B, K)

    loss = (- np.sum(np.log(pos_sig + 1e-9))
            - np.sum(np.log(1.0 - neg_sig + 1e-9)))

    d_pos   = pos_sig - 1.0                                                 # (B,)
    grad_vc = d_pos[:, None] * u_o + np.einsum('bk,bkd->bd', neg_sig, u_n)  # (B, D)
    grad_uo = d_pos[:, None] * v_c                                          # (B, D)
    grad_un = np.einsum('bk,bd->bkd', neg_sig, v_c)                         # (B, K, D)

    # add.at accumulates correctly when the same index appears multiple times
    np.add.at(W_in,  centers,             -lr * grad_vc)
    np.add.at(W_out, contexts,            -lr * grad_uo)
    np.add.at(W_out, neg_samples.ravel(), (-lr * grad_un).reshape(-1, D))

    return float(loss)


def train(data, keep_prob, noise_dist, W_in, W_out, args):
    epoch_len   = len(data)  # original size - LR schedule basis
    total_steps = epoch_len * args.epochs
    vocab_size  = len(noise_dist)
    chunk_size  = args.chunk_size
    batch_size  = args.batch_size
    step        = 0

    for epoch in range(args.epochs):
        epoch_data = apply_subsampling(data, keep_prob)
        n_tokens   = len(epoch_data)
        print(f"Epoch {epoch+1} | {n_tokens:,} tokens after subsampling")

        total_loss    = 0.0
        total_pairs   = 0
        recent_loss   = 0.0
        recent_pairs  = 0
        last_logged   = 0
        t0 = time.time()

        for chunk_start in range(0, n_tokens, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_tokens)

            pad_l  = min(chunk_start, args.window)
            pad_r  = min(n_tokens - chunk_end, args.window)
            padded = epoch_data[chunk_start - pad_l : chunk_end + pad_r]

            centers, contexts = generate_pairs(
                padded, args.window,
                center_start=pad_l,
                center_end=pad_l + (chunk_end - chunk_start),
            )
            n_pairs = len(centers)
            if n_pairs == 0:
                continue

            neg_samples = np.random.choice(
                vocab_size, size=(n_pairs, args.neg_k), p=noise_dist
            ).astype(np.int32)

            mid      = chunk_start + (chunk_end - chunk_start) // 2
            progress = (epoch * epoch_len + mid) / total_steps
            lr       = max(args.lr_init * (1 - progress), 0.0001)

            chunk_loss = 0.0
            for b in range(0, n_pairs, batch_size):
                b_end       = min(b + batch_size, n_pairs)
                chunk_loss += train_step_batch(
                    centers[b:b_end], contexts[b:b_end],
                    neg_samples[b:b_end], lr, W_in, W_out,
                )
                step += b_end - b

            total_loss   += chunk_loss
            total_pairs  += n_pairs
            recent_loss  += chunk_loss
            recent_pairs += n_pairs

            if chunk_end - last_logged >= args.log_interval or chunk_end == n_tokens:
                elapsed    = time.time() - t0
                epoch_avg  = total_loss  / max(total_pairs,  1)
                recent_avg = recent_loss / max(recent_pairs, 1)
                print(f"Epoch {epoch+1} | Token {chunk_end:,}/{n_tokens:,} | "
                      f"Loss (epoch avg): {epoch_avg:.4f} | "
                      f"Loss (recent): {recent_avg:.4f} | "
                      f"LR: {lr:.5f} | Time: {elapsed:.1f}s")
                recent_loss  = 0.0
                recent_pairs = 0
                last_logged  = chunk_end

            if args.checkpoint_interval > 0:
                if (chunk_end // args.checkpoint_interval) > (chunk_start // args.checkpoint_interval):
                    meta = {
                        "epoch": epoch + 1,
                        "token": chunk_end,
                        "step":  step,
                        "loss_epoch_avg": total_loss / max(total_pairs, 1),
                        "lr":    lr,
                    }
                    save_checkpoint(W_in, W_out, meta, args.checkpoint_dir)

        print(f"Epoch {epoch+1} complete. "
              f"Avg pair loss: {total_loss / max(total_pairs, 1):.4f}")

    print("Training complete.")


def cosine_similarity(v, M):
    norms = np.linalg.norm(M, axis=1) + 1e-9
    return (M @ v) / (norms * (np.linalg.norm(v) + 1e-9))


def nearest_neighbours(word, word_to_idx, idx_to_word, W_in, top_n=10):
    if word not in word_to_idx:
        print(f"'{word}' not in vocab.")
        return
    idx  = word_to_idx[word]
    vec  = W_in[idx]
    sims = cosine_similarity(vec, W_in)
    sims[idx] = -1
    top_idx = np.argsort(sims)[::-1][:top_n]
    print(f"\nNearest neighbours to '{word}':")
    for i in top_idx:
        print(f"  {idx_to_word[i]:<20} {sims[i]:.4f}")
