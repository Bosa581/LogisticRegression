# Logistic Regression Sentiment Classifier

A logistic regression sentiment classifier implemented from scratch in Python. The model classifies positive and negative text using stochastic gradient descent and supports raw word-count features, TF-IDF features, L2 regularization, lazy regularization, and configurable learning rates.

## Methods

- `sigmoid`: Converts a score to a probability while preventing exponential overflow.
- `Example`: Builds feature vectors from word counts and optionally computes normalized TF-IDF features.
- `LogReg.progress`: Calculates log probability and classification accuracy.
- `LogReg.sg_update`: Performs a stochastic-gradient update, with optional TF-IDF and lazy regularization.
- `LogReg.finalize_lazy`: Applies pending regularization updates after lazy training.
- `read_dataset`: Loads positive and negative examples, creates train/test splits, and shuffles them.

## Usage

```bash
python logreg.py
```

Optional arguments can configure the training step size, number of passes, regularization, input files, and extra-credit modes.

## Tests

Run the unit tests with:

```bash
python tests.py
```

The tests check gradient updates, TF-IDF feature values, and model evaluation behavior.
