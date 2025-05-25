import numpy as np
import pandas as pd
from utils import normalize_data, train_test_split, accuracy_score
from naive_bayes import NaiveBayes
from sklearn.metrics import classification_report

# Đọc dữ liệu Cleveland (14 thuộc tính, phân tách bằng dấu phẩy, giá trị thiếu là '?')
col_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df = pd.read_csv("D:\Wed_basic\Python\Mining\data-mining-project\data\processed.cleveland.data", names=col_names)

# Xử lý giá trị thiếu
df = df.replace('?', np.nan)
df = df.dropna()
df = df.astype(float)

# Chuyển nhãn về 0 (không bệnh) và 1 (có bệnh)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

X = df.drop('target', axis=1).values
y = df['target'].values

# Chuẩn hóa dữ liệu
X = normalize_data(X)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện và đánh giá Naive Bayes
nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Naive Bayes accuracy:", acc)
print("Số mẫu test:", len(y_test))
print("Số dự đoán đúng:", np.sum(y_pred == y_test))
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=["Không bệnh", "Có bệnh"]))

# Dự đoán cho 1 người bệnh nhập từ bàn phím
print("\nNhập thông tin bệnh nhân để kiểm tra dự đoán bệnh tim:")

feature_names = df.drop('target', axis=1).columns.tolist()
input_values = []
for fname in feature_names:
    while True:
        val = input(f"{fname}: ")
        try:
            val = float(val)
            input_values.append(val)
            break
        except ValueError:
            print("Vui lòng nhập một số hợp lệ!")

# Chuẩn hóa dữ liệu nhập vào (dùng thông số từ toàn bộ X)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
input_norm = (np.array(input_values) - mean) / std
input_norm = input_norm.reshape(1, -1)

# Dự đoán với Naive Bayes
pred = nb.predict(input_norm)[0]
print("\nKết quả dự đoán cho bệnh nhân:")
print("Có bệnh tim" if pred == 1 else "Không bị bệnh tim")