# PRODIGY_DS_03 - Bank Marketing: Customer Lead Scoring

Data Science Internship project (Prodigy InfoTech, Task 03).
A Decision Tree model that predicts whether a bank customer is likely to
subscribe to a term deposit, plus a small console tool that scores a customer.

## Real-World Problem

Banks run telephone campaigns to sell term deposits. Calling every customer is
slow and expensive, and most customers say no. If the bank can tell in advance
which customers are more likely to say yes, its team can spend its calls where
they matter most.

## Objectives

1. Load and check the Bank Marketing dataset.
2. Explore how subscription differs by customer details.
3. Train a Decision Tree Classifier and improve it (limit its depth, handle the imbalance).
4. Evaluate it with accuracy, precision, recall and F1, not accuracy alone.
5. Build a simple tool that scores a new customer.

## Dataset

Bank Marketing dataset (UCI Machine Learning Repository):
https://archive.ics.uci.edu/dataset/222/bank+marketing

- File used: `bank.csv` (separator `;`)
- 4,521 customers, 17 columns (16 inputs and the target `y`)
- No missing values and no duplicate rows
- Target `y`: 521 customers subscribed (yes), 4,000 did not (no), so only 11.5% said yes

## Technologies

Python, pandas, Matplotlib, Seaborn, scikit-learn, joblib (installed with scikit-learn), Git and GitHub.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Checks the data, draws charts, trains and compares models, saves the final model |
| `predict.py` | Console tool that scores a customer using the saved model |
| `model.joblib` | The trained model and the encoders used by `predict.py` |
| `outputs/` | Charts and the model comparison table |
| `bank.csv` | The dataset |
| `requirements.txt` | Libraries to install |

## How to Run

1. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
2. Train and evaluate (creates `outputs/` and `model.joblib`):
   ```
   python train_model.py
   ```
3. Score customers:
   ```
   python predict.py demo      # runs 3 sample customers
   python predict.py           # asks you for a customer's details
   ```

If your computer blocks the libraries, the same commands work in Google Colab
(put each command after `!`).

## Method

1. **Check the data:** missing values, duplicates and the yes/no balance.
2. **Explore:** subscription rate by job and by the outcome of the previous campaign.
3. **Prepare:** convert text columns to numbers with `LabelEncoder`. The column
   `duration` (length of the last call) is **left out**, because it is only known
   after the customer has been called, so it cannot be used to decide whom to call.
4. **Split:** 80% training and 20% test, keeping the same yes/no ratio in both parts.
5. **Train and compare:** an "always no" baseline, the original Decision Tree,
   a Decision Tree with a depth limit and `class_weight="balanced"`, and a Random Forest.
   The tree depth is chosen by 5-fold cross-validation on the training data only.
6. **Evaluate:** accuracy, precision, recall and F1 for the "yes" class, plus a confusion matrix.

## Results (test set, 905 customers)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Always predict no | 0.885 | 0.000 | 0.000 | 0.000 |
| Decision Tree (original) | 0.817 | 0.250 | 0.298 | 0.272 |
| **Decision Tree (pruned, balanced)** | 0.764 | 0.228 | 0.442 | 0.301 |
| Random Forest (balanced) | 0.841 | 0.315 | 0.327 | 0.321 |

Random Forest numbers can differ slightly between computers and library versions.

What this means for the bank:

- The final tree flags 22.3% of customers as worth calling.
- Of those flagged, 22.8% subscribe, compared with 11.5% when calling at random, about twice as good.
- It finds 44.2% of all customers who would subscribe.
- Its accuracy (76.4%) is lower than the "always no" baseline (88.5%). That is expected on this
  dataset: the baseline never finds a single subscriber, so accuracy alone is a poor measure here.

## Limitations

- The model is modest. Even the best model catches under half of the subscribers.
- The dataset is small (4,521 customers) and imbalanced.
- Results come from a single train/test split.
- The model was not deployed and has not been tested on a real bank's customers.
- `predict.py` is a simple console tool, not a full application.

## Future Scope

- Try other models and tune them more carefully.
- Use the larger version of the Bank Marketing dataset.
- Add more evaluation, such as ROC-AUC and k-fold cross-validation on the final model.
- Build a simple web or spreadsheet interface for the bank's staff.
