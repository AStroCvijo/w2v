# word2vec

Skip-gram with negative sampling, implemented from scratch in NumPy. Trains on the text8 corpus and evaluates on the Google analogy and WordSim-353 benchmarks.

## Setup

```bash
conda create -n w2v python=3.9
conda activate w2v
pip install numpy kagglehub
```

The training corpus (text8) is downloaded automatically via `kagglehub` on the first run.

## Benchmark files

Download once and place in the project root:

```bash
# Google analogy (19,544 questions — semantic + syntactic)
wget https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt

# WordSim-353 (353 word pairs with human similarity ratings)
wget https://raw.githubusercontent.com/commonsense/conceptnet5/master/conceptnet5/support_data/wordsim-353/combined.tab
```

## Train

```bash
python main.py --epochs 5 --embed-dim 200
```

Embeddings are saved to `embeddings/` after training. Checkpoints are written to `embeddings/checkpoints/` every 500k tokens.

Key options:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 5 | Training passes over the corpus |
| `--embed-dim` | 200 | Embedding dimensionality |
| `--window` | 5 | Max context window radius |
| `--neg-k` | 5 | Negative samples per positive pair |
| `--lr-init` | 0.025 | Initial learning rate |
| `--min-count` | 5 | Min word frequency for vocabulary |
| `--max-vocab` | 30000 | Vocabulary size cap |

## Eval

### Interactive REPL

Inspect embeddings after training:

```bash
python eval.py
```

```
> nn king                      # nearest neighbours
> king - man + woman           # vector arithmetic
> paris - france + germany     # analogy
> quit
```

Words used in an expression are excluded from the results so they don't trivially dominate.

To inspect a checkpoint instead of the final embeddings:

```bash
python eval.py --w-in-path embeddings/checkpoints/e02_t0003000000/W_in.npy
```

### Benchmarks

```bash
# Analogy accuracy (semantic + syntactic breakdown)
python eval.py --benchmark-analogy questions-words.txt

# Word similarity Spearman correlation
python eval.py --benchmark-similarity combined.tab

# Both at once
python eval.py --benchmark-analogy questions-words.txt --benchmark-similarity combined.tab
```

Results on text8 across configurations:

| epochs | embed-dim | neg-k | Analogy % | WordSim ρ |
|--------|-----------|-------|-----------|-----------|
| 5      | 200       | 5     | 29.2%     | 0.700     |
| 5      | 200       | 10    | 32.2%     | 0.706     |
| 5      | 300       | 5     | 29.2%     | 0.701     |
| 10     | 200       | 5     | 37.8%     | 0.731     |
| 10     | 200       | 10    | **42.7%** | 0.720     |
| 10     | 300       | 10    | 39.3%     | **0.731** |
