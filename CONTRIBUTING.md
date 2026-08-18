# Contributing

Thank you for helping improve ProtFunc ML.

1. Fork the repository and create a focused branch.
2. Install development dependencies with `python -m pip install -e ".[app,dev]"`.
3. Add tests for behavior changes.
4. Run `ruff check .` and `python -m unittest discover -s tests -v`.
5. Open a pull request explaining the motivation, approach, and evaluation impact.

For model changes, report the dataset snapshot, split method, random seed, class counts,
and both macro F1 and balanced accuracy. Do not commit large datasets, trained model
binaries, credentials, private sequences, or personally identifying data.

