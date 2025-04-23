# N-gram Language Modeling Project

This project implements and evaluates unigram and bigram language models using the Penn Treebank (PTB) dataset. It includes preprocessing steps, Laplace (add-alpha) smoothing for bigram models, and perplexity calculation for model evaluation.

## Features

*   Loads and preprocesses the PTB dataset.
*   Adds `<stop>` tokens to sentences.
*   Filters out short sentences.
*   Identifies and replaces rare words with `<unk>`.
*   Implements Unigram language model estimation.
*   Implements Bigram language model estimation.
*   Implements Add-Alpha (Laplace) smoothing for Bigram models.
*   Calculates sentence log-probabilities for both models.
*   Calculates perplexity to evaluate model performance.

## Requirements

*   `datasets` library
*   `tqdm` library

## Installation

1.  Clone the repository.

2.  Install the required libraries:
    ````bash
    uv sync
    ````

## Usage
Run the main script to train and evaluate the models:
```bash
uv run src/main.py
```
### The script will:

1. Download and load the PTB dataset.

2. Preprocess the data (add stop tokens, filter short sentences, handle rare words).

3. Split the data into training and testing sets.

4. Train a Unigram model and evaluate its perplexity.

5. Train a Bigram model (unsmoothed) and report statistics on zero-probability sentences.

6. Train a smoothed Bigram model (Laplace) and evaluate its perplexity.

7. Print log probabilities for example sentences and perplexity scores to the console.

## File Structure

* [main.py](/src/main.py): The main script to run the project. Handles data loading, preprocessing pipeline, model training, and evaluation calls.

* [ngram.py](/src/ngram.py): Contains the core logic for estimating unigram and bigram models (including smoothed), calculating sentence log-probabilities, and computing perplexity.

* [preprocessing.py](/src/preprocessing.py): Contains functions for various data preprocessing steps like adding stop tokens, filtering, and handling rare words.

## Implementation Details

* **Preprocessing:** Uses the `datasets` library for efficient data handling and mapping operations. Rare words (frequency < `THRESHOLD` in the training set) are replaced by `<unk>`.

* **N-gram Models:** Implemented using `collections.defaultdict` for efficient counting and probability storage.

* **Smoothing:** Add-Alpha smoothing is applied to the bigram model to handle unseen word pairs.
Evaluation: Perplexity is calculated based on the negative log-likelihood of the test set, normalized by the total number of words.

## Side Notes
* This is a project from Prof. Osendorfer's NLP course at the HAW Landshut. 
* The project is a learning exercise and is not intended for production use.
* It was very helpful and fun to implement the n-gram models and understand the underlying concepts of language modeling. (Thanks, Prof. Osendorfer!)