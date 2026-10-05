"""Compatibility entry point for the corrected TF-IDF retrieval evaluation.

The historical output in results/retrieval_eval_results.json is preserved.
Use python -m experiments.run for embeddings, generation, and other methods.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run import main

if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--retrieval-only', '--methods', 'tfidf',
                '--output', str(ROOT / 'results/retrieval_tfidf_corrected.json'), *sys.argv[1:]]
    main()
