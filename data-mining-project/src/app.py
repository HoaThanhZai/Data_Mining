import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from utils import normalize_data
from naive_bayes import NaiveBayes
from decision_tree import build_tree, predict

app = Flask(__name__)

# Đọc và xử lý dữ liệu, huấn luyện mô hình
col_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df = pd.read_csv("../data/processed.cleveland.data", names=col_names)
df = df.replace('?', np.nan).dropna().astype(float)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
X = df.drop('target', axis=1).values
y = df['target'].values
X_norm = normalize_data(X)

# Huấn luyện mô hình
nb = NaiveBayes()
nb.fit(X_norm, y)
feature_names = df.drop('target', axis=1).columns.tolist()
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

# Huấn luyện cây quyết định
tree = build_tree(X_norm, y, max_depth=5)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None
    selected_algo = "naive_bayes"
    if request.method == "POST":
        try:
            input_values = [float(request.form[f]) for f in feature_names]
            input_norm = (np.array(input_values) - mean) / std
            input_norm = input_norm.reshape(1, -1)
            selected_algo = request.form.get("algorithm", "naive_bayes")
            if selected_algo == "decision_tree":
                pred = predict(tree, input_norm)[0]
                # Tính score trên toàn bộ dữ liệu
                y_pred_all = predict(tree, X_norm)
                score = np.mean(y_pred_all == y)
            else:
                pred = nb.predict(input_norm)[0]
                score = nb.score(X_norm, y)
            result = "Có bệnh tim" if pred == 1 else "Không bị bệnh tim"
        except Exception as e:
            result = f"Lỗi nhập dữ liệu: {e}"
    return render_template("form.html", feature_names=feature_names, result=result, score=score, selected_algo=selected_algo)

if __name__ == "__main__":
    app.run(debug=True)