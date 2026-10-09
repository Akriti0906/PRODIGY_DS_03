# PRODIGY_DS_03 - Customer Lead Scorer
# Enter a customer's details and the model says whether the customer
# is worth calling about a term deposit.
#
# Run train_model.py first (it creates model.joblib).
#   python predict.py                         -> asks you for a customer's details
#   python predict.py demo                    -> runs 3 sample customers
#   python predict.py rank customers.csv      -> ranks a whole file of customers
#                                                (same columns as bank.csv, ; separated)

import os
import sys
import joblib
import pandas as pd

if not os.path.exists("model.joblib"):
    print("model.joblib not found. Run 'python train_model.py' first to create it.")
    sys.exit(1)

saved = joblib.load("model.joblib")
model = saved["model"]
encoders = saved["encoders"]
features = saved["features"]

# Default answers (press Enter to accept them)
defaults = {
    "age": 40, "job": "management", "marital": "married", "education": "secondary",
    "default": "no", "balance": 1000, "housing": "yes", "loan": "no",
    "contact": "cellular", "day": 15, "month": "may", "campaign": 2,
    "pdays": -1, "previous": 0, "poutcome": "unknown",
}

questions = {
    "age": "Age in years",
    "job": "Job",
    "marital": "Marital status",
    "education": "Education",
    "default": "Has credit in default?",
    "balance": "Average yearly balance (euros)",
    "housing": "Has a housing loan?",
    "loan": "Has a personal loan?",
    "contact": "Contact type",
    "day": "Day of month of last contact (1-31)",
    "month": "Month of last contact (jan, feb, ...)",
    "campaign": "Number of contacts in this campaign",
    "pdays": "Days since last campaign contact (-1 = never contacted)",
    "previous": "Number of contacts before this campaign",
    "poutcome": "Outcome of previous campaign",
}


def ask_number(column):
    while True:
        text = input(f"{questions[column]} [{defaults[column]}]: ").strip()
        if text == "":
            return defaults[column]
        try:
            return int(text)
        except ValueError:
            print("  Please enter a whole number.")


def ask_choice(column):
    options = list(encoders[column].classes_)
    print(f"{questions[column]}")
    print("  Options:", ", ".join(options))
    while True:
        text = input(f"  Your answer [{defaults[column]}]: ").strip().lower()
        if text == "":
            return defaults[column]
        if text in options:
            return text
        print("  Please type one of the options shown above.")


def score_customer(customer):
    row = dict(customer)
    for column in encoders:                 # turn text answers into numbers
        if column in row:
            row[column] = encoders[column].transform([row[column]])[0]
    X = pd.DataFrame([row])[features]       # same column order as training
    prediction = model.predict(X)[0]
    score = model.predict_proba(X)[0][1]
    return prediction, score


def show_result(prediction, score):
    print("\n----------------------------------------")
    if prediction == 1:
        print("Result: WORTH CALLING (likely to subscribe)")
    else:
        print("Result: NOT A PRIORITY (unlikely to subscribe)")
    print(f"Model score for 'yes': {score:.2f}")
    print("Note: this is a rough guide from a small dataset, not a guarantee.")
    print("----------------------------------------")


def run_demo():
    samples = {
        "Customer A (contacted before, previous campaign succeeded)":
            {**defaults, "age": 35, "job": "management", "poutcome": "success",
             "pdays": 90, "previous": 2, "balance": 2500, "month": "mar"},
        "Customer B (new customer, has loans, small balance)":
            {**defaults, "age": 29, "job": "blue-collar", "housing": "yes",
             "loan": "yes", "balance": 50, "contact": "cellular", "campaign": 4},
        "Customer C (retired, no loans, good balance)":
            {**defaults, "age": 66, "job": "retired", "housing": "no",
             "loan": "no", "balance": 3000, "month": "oct"},
    }
    for name, customer in samples.items():
        print("\n" + name)
        prediction, score = score_customer(customer)
        show_result(prediction, score)


def run_interactive():
    print("Enter the customer's details (press Enter to accept the default).\n")
    customer = {}
    for column in features:
        if column in encoders:
            customer[column] = ask_choice(column)
        else:
            customer[column] = ask_number(column)
    prediction, score = score_customer(customer)
    show_result(prediction, score)


def run_rank(filename):
    """Score every customer in a file and list them from most to least likely."""
    if not os.path.exists(filename):
        print(f"File '{filename}' not found.")
        sys.exit(1)
    customers = pd.read_csv(filename, sep=";")
    missing = [c for c in features if c not in customers.columns]
    if missing:
        print("The file is missing these columns:", ", ".join(missing))
        sys.exit(1)

    X = customers[features].copy()
    for column in encoders:
        if column in X.columns:
            unknown = set(X[column]) - set(encoders[column].classes_)
            if unknown:
                print(f"Column '{column}' has values the model has not seen: {sorted(unknown)}")
                sys.exit(1)
            X[column] = encoders[column].transform(X[column])

    customers["score"] = model.predict_proba(X)[:, 1].round(3)
    customers["decision"] = ["CALL" if p == 1 else "SKIP" for p in model.predict(X)]
    ranked = customers.sort_values("score", ascending=False)   # best leads first
    ranked.to_csv("ranked_customers.csv", index=False, sep=";")

    print(f"Scored {len(ranked)} customers. Customers marked CALL: "
          f"{(ranked['decision'] == 'CALL').sum()}")
    print("\nTop 10 customers to call first:")
    print(ranked[["age", "job", "poutcome", "score", "decision"]].head(10).to_string())
    print("\nFull ranked list saved as ranked_customers.csv")
    print("Note: many customers share the same score because a tree groups them into a few types.")


if len(sys.argv) > 1 and sys.argv[1] == "demo":
    run_demo()
elif len(sys.argv) > 2 and sys.argv[1] == "rank":
    run_rank(sys.argv[2])
else:
    run_interactive()
