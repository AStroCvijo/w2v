import json
import os
import numpy as np


def _atomic_npy(path, array):
    # must end in .npy or numpy appends it, breaking os.replace
    tmp = path[:-4] + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def _atomic_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else x)
    os.replace(tmp, path)


def save_checkpoint(W_in, W_out, meta, checkpoint_dir):
    name    = f"e{meta['epoch']:02d}_t{meta['token']:010d}"
    ckpt_dir = os.path.join(checkpoint_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    _atomic_npy(os.path.join(ckpt_dir, "W_in.npy"),   W_in)
    _atomic_npy(os.path.join(ckpt_dir, "W_out.npy"),  W_out)
    _atomic_json(os.path.join(ckpt_dir, "meta.json"), meta)
    print(f"  [checkpoint → {ckpt_dir}]")


def save_final(W_in, W_out, idx_to_word, vocab_words, args):
    out_dir = os.path.dirname(args.w_in_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _atomic_npy(args.w_in_path,  W_in)
    _atomic_npy(args.w_out_path, W_out)
    tmp = args.vocab_path + ".tmp"
    with open(tmp, "w") as f:
        for i in range(len(vocab_words)):
            f.write(idx_to_word[i] + "\n")
    os.replace(tmp, args.vocab_path)
    print(f"\nFinal embeddings saved → {out_dir or '.'}")
