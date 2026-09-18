# GMiS-Data-Analytics-Challenge-2026
Team Repository for GMiS Data Analytics Challenge held on 2026-09-19

# Network Traffic Classifier (LightGBM Multiclass)

This script trains a multiclass [LightGBM](https://lightgbm.readthedocs.io/) classifier on a labeled dataset (`D1`) to recognize network traffic/attack categories (e.g., benign traffic vs. various DDoS/DoS/exploit/scan types), then uses the trained model to predict labels for an unlabeled dataset (`D2`).

## What the script does

1. **Loads and cleans two pickled DataFrames** (`D1.pkl` = labeled training data, `D2.pkl` = unlabeled data to predict).
   - Strips column name whitespace.
   - Detects numeric-looking text columns (e.g., `"51,236"`) and converts them to floats.
   - Leaves genuinely categorical columns (e.g., `Protocol`) as uppercase strings.
   - Replaces `inf`/`-inf` with `NaN` and downcasts numeric columns to `float32`.

2. **Normalizes and validates labels in D1.**
   - Lowercases/trims label text and standardizes dash characters.
   - Drops duplicate rows.
   - Prints class counts, warns about classes present in the fixed `CLASS_NAMES` list but missing from the data, warns about unrecognized labels, and drops classes with fewer than `MIN_ROWS_PER_CLASS` rows.

3. **Aligns features between D1 and D2.**
   - Confirms D2 contains every feature column used from D1 (raises an error if not).
   - Builds a shared category list per categorical column so D1 and D2 use identical category encodings.

4. **Splits and trains the model.**
   - Stratified 80/20 train/validation split.
   - Optional per-class row cap (`CAP_PER_CLASS`) applied only to the training set, to limit the influence of very large classes.
   - Trains an `LGBMClassifier` (multiclass objective) with early stopping on the validation set.

5. **Evaluates on the validation split.**
   - Prints macro-F1, a full classification report, and saves a confusion matrix to `confusion_matrix.csv`.
   - Prints the top 15 most-confused class pairs (true label → predicted label).

6. **Retrains a final model on all of D1** (using the best iteration count found during validation) and predicts on D2.
   - Adds predicted `Label`, `LabelID` (mapped via a fixed class list), `Confidence` (max predicted probability), and `LowConfidence` (flag for predictions below `CONF_THRESHOLD`) columns to D2.
   - Saves the result to `D2_labeled.csv`.
   - Writes `output.txt`, one predicted class ID per D2 row (raises an error if any predicted label isn't in the class-ID mapping).

## Inputs / Outputs

| File | Direction | Description |
|---|---|---|
| `D1.pkl` | input | Labeled training dataset (pickle) |
| `D2.pkl` | input | Unlabeled dataset to classify (pickle) |
| `confusion_matrix.csv` | output | Validation confusion matrix |
| `D2_labeled.csv` | output | D2 with predicted label, label ID, confidence, low-confidence flag |
| `output.txt` | output | One predicted class ID per line for D2 rows |

## Key configuration (top of script)

- `D1_PKL`, `D2_PKL`, `OUT_CSV` — input/output file paths
- `LABEL` — name of the label column
- `CAP_PER_CLASS` — max training rows per class (`None` disables capping)
- `MIN_ROWS_PER_CLASS` — classes below this row count are dropped
- `USE_BALANCED_WEIGHTS` — use class-balanced weighting (helps rare-class recall)
- `CONF_THRESHOLD` — confidence cutoff below which D2 predictions are flagged
- `DROP_COLUMNS` — columns to exclude from features (e.g., IPs, timestamps, index columns)
- `SEED` — random seed for reproducibility
- `CLASS_NAMES` — fixed, ordered list of all recognized traffic/attack classes, used to build the `CLASS_ID` mapping

## Dependencies / Libraries

- [`numpy`](https://numpy.org/)
- [`pandas`](https://pandas.pydata.org/)
- [`lightgbm`](https://lightgbm.readthedocs.io/) (`lgb.LGBMClassifier`, `lgb.early_stopping`, `lgb.log_evaluation`)
- [`scikit-learn`](https://scikit-learn.org/) — specifically:
  - `sklearn.model_selection.train_test_split`
  - `sklearn.preprocessing.LabelEncoder`
  - `sklearn.metrics.classification_report`, `f1_score`, `confusion_matrix`

Install with:

```bash
pip install numpy pandas lightgbm scikit-learn
```

## Usage

1. Place `D1.pkl` (labeled) and `D2.pkl` (unlabeled) in the working directory, or update the paths in the settings block.
2. Adjust `DROP_COLUMNS`, `CAP_PER_CLASS`, `USE_BALANCED_WEIGHTS`, and `CONF_THRESHOLD` as needed for your dataset.
3. Run:

```bash
python main.py
```

4. Review console output (class counts, macro-F1, classification report, top confused pairs) and the generated `confusion_matrix.csv`, `D2_labeled.csv`, and `output.txt`.

## Notes

- The script expects D1 and D2 to share the same feature columns aside from the label; it will raise a `ValueError` if D2 is missing any required column.
- It will also raise a `ValueError` at the end if any predicted D2 label falls outside the fixed `CLASS_ID` mapping — this guards against silently writing invalid IDs to `output.txt`.
- Rows with low-confidence predictions (`Confidence < CONF_THRESHOLD`) are flagged, not removed, so they can be reviewed manually.
