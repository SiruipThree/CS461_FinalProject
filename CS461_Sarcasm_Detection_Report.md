# Sarcasm Detection in News Headlines

**Portfolio report author:** Sirui Peng\
**Academic context:** Rutgers University, CS461 — Fall 2025\
**Original project period:** December 2025\
**Documentation and saved-model verification:** October 4, 2026

[Project overview](README.md) · [Training notebook](SarcasmDetection.ipynb) · [Inference implementation](predict_sarcasm.py) · [Verification record](docs/verification-2026-10-04.json)

## 1. Introduction

This project classifies news headlines as sarcastic (`1`) or non-sarcastic (`0`). Short headlines offer limited context, and similar words can appear in both classes. The modeling approach therefore compares statistical text features, word sequences, and character sequences.

I implemented data exploration and preprocessing, a TF-IDF Logistic Regression baseline, a randomly initialized BiLSTM, a BiLSTM initialized with GloVe embeddings, a character-level CNN, and a Logistic Regression stacking model. I also implemented a command-line inference pipeline using saved model weights and preprocessing artifacts.

The final inference architecture combines probabilities from the GloVe BiLSTM and character CNN. On October 4, 2026, the saved artifacts were evaluated through the existing CLI on the supplied 966-row test split, producing **87.99% accuracy** and **0.8731 F1** for the sarcastic class.

The original notebook retains course-era team credits for Sirui Peng and Henry You. This portfolio write-up describes my implementation work and the evidence available in the repository.

## 2. Data Exploration and Preprocessing

### 2.1 Supplied data

Each CSV has `text` and `label` columns. Counts below were checked directly against the repository files.

| Split | Headlines | Non-sarcastic (`0`) | Sarcastic (`1`) |
|---|---:|---:|---:|
| Training | 21,464 | 11,248 | 10,216 |
| Validation | 716 | 360 | 356 |
| Test | 966 | 526 | 440 |
| **Total** | **23,146** | **12,134** | **11,012** |

The training data is reasonably balanced, with approximately 47.6% sarcastic labels. Inspection of the supplied files found no empty headline strings.

### 2.2 Model-specific preprocessing

The TF-IDF baseline explicitly lowercases headlines, fits its vectorizer on training text, and transforms validation and test text with the same fitted vectorizer.

The neural models use Keras tokenizers fitted on the training text:

- **Word sequences:** vocabulary cap of 10,000, an out-of-vocabulary token, and a maximum length of 100 positions.
- **Character sequences:** vocabulary cap of 100, character-level tokenization, and a maximum length of 300 positions.
- **Padding and truncation:** both are applied at the end of each sequence (`post`).

The tokenizers use their default lowercase behavior. The implementation does not add a separate stemming, lemmatization, stopword-removal, or augmentation stage. The GloVe embedding matrix begins with zeros; words found in the pretrained file receive their corresponding vectors before training.

For inference, the script loads the saved tokenizers, fills missing input text with an empty string, converts values to strings, and applies the saved length configuration. This keeps the inference transformations aligned with the saved models.

### 2.3 Data-quality observations

An exact-text check found 74 duplicate text rows within training and one within test. There are three distinct headlines shared by training and validation, and five shared by training and test; validation and test share none. The evaluation uses the supplied splits as they are. These observations matter when interpreting generalization and motivate deduplication before a future training cycle.

## 3. Feature Engineering

### 3.1 TF-IDF baseline

The baseline uses `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))`. Unigrams capture individual words, while bigrams capture short combinations. A Logistic Regression classifier is fitted on these sparse features with `max_iter=1000` and `random_state=42`.

This provides a simple reference point for evaluating the sequence models.

### 3.2 Word embeddings

The basic BiLSTM learns 128-dimensional embeddings from random initialization. The GloVe variant uses pretrained 100-dimensional vectors from `glove.6B.100d.txt` to initialize its embedding layer.

**The GloVe embedding layer is trainable.** The notebook creates the model with `trainable=True`, and the saved model's embedding configuration also has this setting. The project therefore performs task-specific fine-tuning of pretrained word embeddings alongside training the recurrent and classification layers.

### 3.3 Character features

The character CNN learns 64-dimensional character embeddings and applies one-dimensional convolutions. This provides a different representation from the word-level model, including local spelling and character patterns.

### 3.4 Stacking features

For each headline, the final meta-learner receives two features in this order:

```text
[character_CNN_probability, GloVe_BiLSTM_probability]
```

The Logistic Regression stacker learns how to combine these scores into the final binary prediction.

## 4. Model Architecture and Selection

### 4.1 Basic BiLSTM

The basic sequence model has a trainable word embedding layer, spatial dropout, two bidirectional LSTM layers with 64 and 32 units, a 64-unit dense layer, dropout, and a sigmoid output. This experiment tests learned word-sequence representations without pretrained embeddings.

### 4.2 BiLSTM initialized with GloVe

The saved GloVe model has the following architecture, confirmed against its serialized configuration:

1. Trainable 100-dimensional embedding layer with a 10,000-word cap.
2. Spatial dropout of 0.2.
3. Bidirectional LSTM with 64 units, sequence output, dropout of 0.2, and recurrent dropout of 0.2.
4. Bidirectional LSTM with 32 units, dropout of 0.2, and recurrent dropout of 0.2.
5. Dense layer with 64 ReLU units.
6. Dropout of 0.3.
7. One sigmoid output unit.

The architecture uses recurrent sequence modeling. It contains no attention layer.

### 4.3 Character CNN

The saved character CNN contains:

1. A 100-entry character embedding layer with 64 dimensions and dropout of 0.2.
2. A 128-filter convolution with kernel size 3, max pooling, and dropout of 0.3.
3. A second 128-filter convolution with kernel size 3, max pooling, and dropout of 0.3.
4. A 64-filter convolution with kernel size 3 and global max pooling.
5. A 64-unit ReLU dense layer with dropout of 0.5.
6. One sigmoid output unit.

The convolutions use the same kernel size; their filter counts and pooling operations define the successive feature stages.

### 4.4 Final ensemble

The final pipeline combines the GloVe BiLSTM and character CNN with a Logistic Regression meta-learner. The word and character branches provide different input representations, while the two-feature stacker keeps the combination step small and inspectable.

The standalone TF-IDF baseline and basic BiLSTM are comparison experiments. The distributed inference script loads the saved GloVe model, character CNN, and stacker.

## 5. Training Methodology

The notebook trains the neural base models with binary cross-entropy and Adam. The GloVe and character CNN training code uses an initial learning rate of 0.001 and a batch size of 64.

| Setting | GloVe BiLSTM | Character CNN |
|---|---|---|
| Maximum epochs | 20 | 30 |
| Early stopping | Validation loss; patience 5; restore best weights | Validation loss; patience 5; restore best weights |
| Learning-rate reduction | Validation loss; factor 0.5; patience 2 | Validation loss; factor 0.5; patience 2 |
| Checkpoint selection | Best validation loss | Best validation loss |

The base models are fitted on the training split. Their validation-set probability predictions are combined into a two-column matrix, and the stacker is fitted on that matrix with the validation labels. Predictions on the supplied test split are then used for evaluation.

The same validation split supports base-model checkpoint selection and stacker fitting. A future evaluation could use a separate stacking split or out-of-fold predictions to better separate these decisions. The current implementation does not establish cross-validation or an independent external benchmark.

The notebook sets NumPy and TensorFlow random seeds for the neural experiments. The historical stored outputs and the saved checkpoints reflect development at different points; fresh training can yield different weights and scores.

## 6. Experiments and Results

### 6.1 Verified saved-model evaluation

On October 4, 2026, the existing `predict_sarcasm.py` was run on all 966 rows of `test.csv`, using the repository's saved models. The generated labels were compared with the CSV's ground-truth labels. This verification ran inference and metric calculation; it did not retrain the models.

| Metric | Result |
|---|---:|
| Accuracy | **0.8799 — 87.99%** |
| Precision, sarcastic class | 0.8418 |
| Recall, sarcastic class | 0.9068 |
| F1, sarcastic class | **0.8731** |

The confusion matrix is:

| Actual / predicted | Non-sarcastic (`0`) | Sarcastic (`1`) |
|---|---:|---:|
| Non-sarcastic (`0`) | 451 | 75 |
| Sarcastic (`1`) | 41 | 399 |

This gives 850 correct predictions, 75 false positives, and 41 false negatives. The generated output matched the repository's `predictions.csv` exactly by text and prediction value.

The [verification record](docs/verification-2026-10-04.json) includes runtime versions, file hashes, metrics, and the confusion matrix so the evaluated artifacts can be identified.

### 6.2 Historical notebook comparisons

These values come from stored outputs in `SarcasmDetection.ipynb`. They describe the recorded development experiments and are separate from the saved-model verification above.

| Model | Test accuracy | Sarcastic-class F1 | Evidence |
|---|---:|---:|---|
| Logistic Regression + TF-IDF | 83.23% | 0.8163 | Baseline evaluation output |
| GloVe BiLSTM | 86.85% | 0.8612 | GloVe evaluation and ensemble comparison outputs |
| Character CNN | 84.58% | 0.8269 | Ensemble comparison output |
| Stacking ensemble | 87.89% | 0.8690 | Final stacking evaluation output |

In that recorded comparison, stacking improved accuracy by about 1.04 percentage points over the GloVe branch and 3.31 points over the character branch. The basic BiLSTM is also implemented in the notebook; different stored summaries give different values for that experiment, so it is excluded from this comparison table.

### 6.3 Result provenance

The repository retains several earlier result artifacts:

- The stored notebook stacking result is 87.89% accuracy.
- `stacking_config.pkl` contains a historical `test_accuracy` value of approximately 89.03%.
- `test_predictions.csv` yields approximately 89.13% when compared with the supplied test labels.
- The verified current CLI output yields **87.99%** and matches `predictions.csv`.

The primary result in this report is the recomputed CLI result. Historical metadata and earlier output files are retained as development records, and their scores are not substituted for the verified saved-model result.

## 7. Discussion

The project demonstrates a complete academic modeling workflow: a statistical baseline, recurrent and convolutional models, task-specific tuning of pretrained embeddings, model combination, and an inference interface that reuses saved preprocessing artifacts.

The verified ensemble has higher recall than precision for the sarcastic class. On the supplied test data, it detects 399 of 440 sarcastic headlines while incorrectly flagging 75 non-sarcastic ones. The appropriate precision-recall balance would depend on the intended application.

Several limits shape interpretation of the results:

- The supplied data has a small number of exact duplicate headlines within and across splits.
- The validation split is used for both neural checkpoint selection and stacker fitting.
- Stored development outputs and saved-artifact metadata are not a single consistent experiment snapshot.
- Generalization to other text domains, latency under production load, and external validation were not measured in this verification.

Useful next steps include deduplicated splits, out-of-fold stacking, explicit experiment manifests linking every metric to a checkpoint, and evaluation on an independently sourced dataset.

## 8. Reproduction and Conclusion

Install the versions in `requirements.txt` in a Python 3.11 environment, then run:

```bash
python predict_sarcasm.py --input test.csv --output /tmp/sarcasm_predictions.csv
```

The script loads six inference artifacts from `models/`: the character CNN, character tokenizer, GloVe BiLSTM, word tokenizer, Logistic Regression meta-learner, and stacking configuration. It validates the input column, prepares both sequence representations, combines base-model probabilities, and writes the final predictions.

To calculate metrics on the generated output:

```bash
python - <<'PY'
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

truth = pd.read_csv('test.csv')
pred = pd.read_csv('/tmp/sarcasm_predictions.csv')
assert truth['text'].equals(pred['text'])
y, y_pred = truth['label'], pred['prediction']
print('Accuracy:', accuracy_score(y, y_pred))
print('Precision:', precision_score(y, y_pred))
print('Recall:', recall_score(y, y_pred))
print('F1:', f1_score(y, y_pred))
PY
```

Saved-model inference was verified with Python 3.11.9, TensorFlow 2.20.0, Keras 3.12.0, and scikit-learn 1.8.0. The raw GloVe download is needed for retraining rather than saved-model inference. The historical notebook contains the training code and experiment outputs; retraining may produce a different result.

The project provides concrete experience with text-feature engineering, neural-network training and evaluation, fine-tuning pretrained word embeddings, ensemble construction, and packaging model inference for reuse.

## 9. References and Repository Evidence

- [Main training notebook](SarcasmDetection.ipynb)
- [Inference implementation](predict_sarcasm.py)
- [Verified saved-model evaluation record](docs/verification-2026-10-04.json)
- [Original course assignment specification](cs_461_final_project.pdf)
- [Stanford GloVe project and pretrained vectors](https://nlp.stanford.edu/projects/glove/)
- [scikit-learn Logistic Regression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [TensorFlow Bidirectional layer documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional)
- [TensorFlow Conv1D layer documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D)
