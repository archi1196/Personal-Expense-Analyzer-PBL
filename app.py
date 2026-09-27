
from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
from datetime import datetime
import json
import uuid

import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "personal-expense-analyzer-pbl"

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "personal_expense_anomaly_dataset_monthwise.csv"
EXCEL_FILE = BASE_DIR / "personal_expense_analyzer.xlsx"
MONEY_FILE = BASE_DIR / "money.json"
BUDGET_FILE = BASE_DIR / "budget.json"
CHART_DIR = BASE_DIR / "static" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "Food", "Transport", "Shopping", "Entertainment",
    "Education", "Bills", "Health", "Other"
]

PAYMENT_MODES = ["Cash", "UPI / Online", "Bank", "Other"]

COLUMNS = [
    "Transaction_ID", "Date", "Category", "Amount",
    "Payment_Mode", "Merchant", "Description"
]


def load_data():
    if not CSV_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_csv(CSV_FILE)

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""

    raw_date = df["Date"].astype(str)
    df["Date"] = pd.to_datetime(raw_date, errors="coerce")
    bad = df["Date"].isna()
    if bad.any():
        df.loc[bad, "Date"] = pd.to_datetime(
            raw_date[bad], errors="coerce", dayfirst=True
        )

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Date", "Amount", "Category"])
    df = df[df["Amount"] > 0].drop_duplicates()
    return df[COLUMNS].sort_values("Date").reset_index(drop=True)


def save_data(df):
    out = df.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(CSV_FILE, index=False)

    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="Expenses")
        summary = pd.DataFrame({
            "Metric": [
                "Total Spending", "Average Expense",
                "Minimum Expense", "Highest Expense",
                "Total Transactions"
            ],
            "Value": [
                df["Amount"].sum(),
                df["Amount"].mean(),
                df["Amount"].min(),
                df["Amount"].max(),
                len(df)
            ]
        })
        summary.to_excel(writer, index=False, sheet_name="Summary")


def get_money():
    defaults = {
        "Cash": 10000.0,
        "UPI / Online": 8500.0,
        "Bank": 50000.0,
        "Other": 2000.0
    }

    if not MONEY_FILE.exists():
        return defaults

    try:
        saved = json.loads(MONEY_FILE.read_text())
        for key in defaults:
            saved[key] = float(saved.get(key, defaults[key]))
        return saved
    except Exception:
        return defaults


def save_money(values):
    MONEY_FILE.write_text(json.dumps(values, indent=2))


def get_budget():
    if not BUDGET_FILE.exists():
        return 20000.0
    try:
        return float(json.loads(BUDGET_FILE.read_text())["budget"])
    except Exception:
        return 20000.0


def set_budget(value):
    BUDGET_FILE.write_text(json.dumps({"budget": value}, indent=2))


def create_charts(df):
    if df.empty:
        return

    category = df.groupby("Category")["Amount"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    category.plot(kind="barh", ax=ax)
    ax.set_title("Category-wise Spending")
    ax.set_xlabel("Amount (₹)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "category.png", dpi=130)
    plt.close(fig)

    monthly = df.set_index("Date").resample("ME")["Amount"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set_title("Monthly Spending Trend")
    ax.set_ylabel("Amount (₹)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "monthly.png", dpi=130)
    plt.close(fig)

    payment = df.groupby("Payment_Mode")["Amount"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    payment.plot(kind="bar", ax=ax)
    ax.set_title("Payment Mode-wise Spending")
    ax.set_ylabel("Amount (₹)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "payment.png", dpi=130)
    plt.close(fig)


def detect_anomalies(df):
    if len(df) < 5:
        result = df.copy()
        result["Prediction"] = 1
        return result

    model = IsolationForest(
        contamination=0.03,
        random_state=42
    )
    result = df.copy()
    result["Prediction"] = model.fit_predict(result[["Amount"]])
    return result


@app.context_processor
def global_data():
    df = load_data()
    money = get_money()
    return {
        "total_spending": df["Amount"].sum() if not df.empty else 0,
        "transaction_count": len(df),
        "money": money,
        "money_total": sum(money.values()),
        "budget_value": get_budget()
    }


@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/dashboard")
def dashboard():
    df = load_data()

    stats = {
        "total": df["Amount"].sum() if not df.empty else 0,
        "average": df["Amount"].mean() if not df.empty else 0,
        "highest": df["Amount"].max() if not df.empty else 0,
        "transactions": len(df)
    }

    return render_template("dashboard.html", stats=stats)


@app.route("/money", methods=["GET", "POST"])
def money():
    values = get_money()

    if request.method == "POST":
        try:
            updated = {
                "Cash": float(request.form["cash"]),
                "UPI / Online": float(request.form["upi"]),
                "Bank": float(request.form["bank"]),
                "Other": float(request.form["other"])
            }

            if any(value < 0 for value in updated.values()):
                raise ValueError

            save_money(updated)
            flash("Money amounts updated successfully.", "success")
            return redirect(url_for("money"))

        except Exception:
            flash("Please enter valid positive amounts.", "error")

    return render_template("money.html", values=values)


@app.route("/add-expense", methods=["GET", "POST"])
def add_expense():
    if request.method == "POST":
        try:
            date = pd.to_datetime(
                request.form["date"], errors="coerce"
            )

            amount = float(request.form["amount"])

            if pd.isna(date) or amount <= 0:
                raise ValueError

            df = load_data()

            new_id = (
                "EXP-" +
                datetime.now().strftime("%Y%m%d%H%M%S") +
                "-" + uuid.uuid4().hex[:4].upper()
            )

            row = pd.DataFrame([{
                "Transaction_ID": new_id,
                "Date": date,
                "Category": request.form["category"],
                "Amount": amount,
                "Payment_Mode": request.form["payment_mode"],
                "Merchant": request.form["merchant"].strip(),
                "Description": request.form["description"].strip()
            }])

            df = pd.concat([df, row], ignore_index=True)
            df = df.drop_duplicates()
            save_data(df)

            flash("Expense added successfully.", "success")
            return redirect(url_for("expenses"))

        except Exception:
            flash("Please enter valid expense details.", "error")

    return render_template(
        "add_expense.html",
        categories=CATEGORIES,
        payment_modes=PAYMENT_MODES
    )


@app.route("/expenses")
def expenses():
    df = load_data()
    rows = df.sort_values("Date", ascending=False).to_dict("records")
    return render_template("expenses.html", rows=rows)


@app.route("/expenses/edit/<transaction_id>", methods=["GET", "POST"])
def edit_expense(transaction_id):
    df = load_data()
    matches = df.index[df["Transaction_ID"].astype(str) == str(transaction_id)].tolist()

    if not matches:
        flash("Expense not found.", "error")
        return redirect(url_for("expenses"))

    index = matches[0]

    if request.method == "POST":
        try:
            date = pd.to_datetime(request.form["date"], errors="coerce")
            amount = float(request.form["amount"])

            if pd.isna(date) or amount <= 0:
                raise ValueError

            df.loc[index, "Date"] = date
            df.loc[index, "Category"] = request.form["category"]
            df.loc[index, "Amount"] = amount
            df.loc[index, "Payment_Mode"] = request.form["payment_mode"]
            df.loc[index, "Merchant"] = request.form["merchant"].strip()
            df.loc[index, "Description"] = request.form["description"].strip()

            save_data(df)
            flash("Expense updated successfully.", "success")
            return redirect(url_for("expenses"))

        except Exception:
            flash("Please enter valid expense details.", "error")

    row = df.loc[index].to_dict()
    row["Date"] = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")

    return render_template(
        "edit_expense.html",
        row=row,
        categories=CATEGORIES,
        payment_modes=PAYMENT_MODES
    )


@app.route("/expenses/delete/<transaction_id>", methods=["POST"])
def delete_expense(transaction_id):
    df = load_data()
    before = len(df)
    df = df[df["Transaction_ID"].astype(str) != str(transaction_id)]

    if len(df) == before:
        flash("Expense not found.", "error")
    else:
        save_data(df)
        flash("Expense deleted successfully.", "success")

    return redirect(url_for("expenses"))


@app.route("/analysis")
def analysis():
    df = load_data()

    if df.empty:
        category = monthly = payment = {}
        stats = {"total": 0, "average": 0, "minimum": 0, "maximum": 0}
    else:
        category = (
            df.groupby("Category")["Amount"]
            .sum().sort_values(ascending=False).to_dict()
        )
        monthly = (
            df.assign(Month=df["Date"].dt.strftime("%B %Y"))
            .groupby("Month")["Amount"].sum().to_dict()
        )
        payment = (
            df.groupby("Payment_Mode")["Amount"]
            .sum().sort_values(ascending=False).to_dict()
        )
        stats = {
            "total": df["Amount"].sum(),
            "average": df["Amount"].mean(),
            "minimum": df["Amount"].min(),
            "maximum": df["Amount"].max()
        }

    return render_template(
        "analysis.html",
        stats=stats,
        category=category,
        monthly=monthly,
        payment=payment
    )


@app.route("/charts")
def charts():
    create_charts(load_data())
    return render_template("charts.html")


@app.route("/anomalies")
def anomalies():
    df = load_data()
    result = detect_anomalies(df)
    found = result[result["Prediction"] == -1].copy()
    found["Date"] = found["Date"].dt.strftime("%Y-%m-%d")

    return render_template(
        "anomalies.html",
        rows=found.to_dict("records"),
        total=len(result),
        anomaly_count=len(found)
    )


@app.route("/budget", methods=["GET", "POST"])
def budget():
    df = load_data()

    if request.method == "POST":
        try:
            value = float(request.form["budget"])
            if value <= 0:
                raise ValueError

            set_budget(value)
            flash("Budget updated successfully.", "success")
            return redirect(url_for("budget"))

        except Exception:
            flash("Enter a valid budget.", "error")

    budget_value = get_budget()

    if df.empty:
        latest_month = "No data"
        spent = 0
    else:
        latest_date = df["Date"].max()
        latest_month = latest_date.strftime("%B %Y")
        spent = df[
            df["Date"].dt.to_period("M") ==
            latest_date.to_period("M")
        ]["Amount"].sum()

    return render_template(
        "budget.html",
        budget=budget_value,
        latest_month=latest_month,
        spent=spent,
        remaining=budget_value - spent
    )


if __name__ == "__main__":
    app.run(debug=True)
