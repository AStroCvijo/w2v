import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Word2Vec — skip-gram with negative sampling")

    # Reproducability
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Data
    parser.add_argument("--min-count",    type=int,   default=5,       help="Minimum word frequency to include in vocabulary")
    parser.add_argument("--max-vocab",    type=int,   default=30_000,  help="Maximum vocabulary size")
    parser.add_argument("--threshold",    type=float, default=1e-5,    help="Subsampling threshold")

    # Model
    parser.add_argument("--embed-dim",    type=int,   default=200,     help="Embedding dimensionality")
    parser.add_argument("--window",       type=int,   default=5,       help="Maximum context window size")
    parser.add_argument("--neg-k",        type=int,   default=5,       help="Number of negative samples per positive pair")
    parser.add_argument("--lr-init",      type=float, default=0.025,   help="Initial learning rate")

    # Training
    parser.add_argument("--epochs",       type=int,   default=5,         help="Number of training epochs")
    parser.add_argument("--batch-size",   type=int,   default=512,       help="Mini-batch size (pairs per gradient step)")
    parser.add_argument("--chunk-size",   type=int,   default=100_000,   help="Tokens per corpus chunk (controls memory vs. Python overhead)")
    parser.add_argument("--log-interval", type=int,   default=100_000,   help="Logging interval in tokens")

    # Checkpointing
    parser.add_argument("--checkpoint-interval", type=int, default=500_000,                  help="Save a checkpoint every N tokens (0 = disabled)")
    parser.add_argument("--checkpoint-dir",      type=str, default="embeddings/checkpoints", help="Directory for training checkpoints")

    # Output
    parser.add_argument("--w-in-path",  type=str, default="embeddings/W_in.npy",  help="Save path for W_in embeddings")
    parser.add_argument("--w-out-path", type=str, default="embeddings/W_out.npy", help="Save path for W_out embeddings")
    parser.add_argument("--vocab-path", type=str, default="embeddings/vocab.txt",  help="Save path for vocabulary")

    return parser.parse_args()
