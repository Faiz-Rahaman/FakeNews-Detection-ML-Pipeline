"""Backward-compatible wrapper for evaluation.

Use:
    python src/test.py ...
This delegates to evaluate.py.
"""

from evaluate import main


if __name__ == "__main__":
    main()
