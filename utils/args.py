import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Word2Vec — skip-gram with negative sampling")

    # Data
    parser.add_argument("--min-count",    type=int,   default=5,       help="Minimum word frequency to include in vocabulary")
    parser.add_argument("--max-vocab",    type=int,   default=30_000,  help="Maximum vocabulary size")
    parser.add_argument("--threshold",    type=float, default=1e-5,    help="Subsampling threshold")

    # Model
    parser.add_argument("--embed-dim",    type=int,   default=100,     help="Embedding dimensionality")
    parser.add_argument("--window",       type=int,   default=5,       help="Maximum context window size")
    parser.add_argument("--neg-k",        type=int,   default=5,       help="Number of negative samples per positive pair")
    parser.add_argument("--lr-init",      type=float, default=0.025,   help="Initial learning rate")

    # Training
    parser.add_argument("--epochs",       type=int,   default=1,       help="Number of training epochs")
    parser.add_argument("--log-interval", type=int,   default=100_000, help="Logging interval in tokens")

    return parser.parse_args()
