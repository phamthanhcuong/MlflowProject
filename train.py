# ---- TRAIN.PY ----

# Import thư viện cần thiết
import mlflow   # MLflow dùng để quản lý việc huấn luyện và lưu mô hình
import mlflow.sklearn  # Dùng MLflow để log mô hình scikit-learn
from sklearn.datasets import make_classification  # Hàm tạo dữ liệu giả cho bài toán phân loại
from sklearn.linear_model import LogisticRegression  # Mô hình hồi quy logistic (phân loại)
from sklearn.model_selection import train_test_split  # Chia dữ liệu thành train/test
from sklearn.metrics import accuracy_score  # Để tính độ chính xác (accuracy)
import pandas as pd  # Thư viện bảng tính dữ liệu

# --- Chuẩn bị dữ liệu ---
# Tạo ra 1000 mẫu dữ liệu, mỗi mẫu có 20 đặc trưng, chia thành 2 nhóm (0 và 1)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
# Chia dữ liệu thành 80% để train, 20% để test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Khai báo các bộ tham số cần thử nghiệm (hyperparameter tuning) ---
param_grid = [
    {"C": 0.01, "solver": "liblinear"},  # Model 1
    {"C": 0.1, "solver": "liblinear"},   # Model 2
    {"C": 1.0, "solver": "lbfgs"},        # Model 3
    {"C": 10.0, "solver": "lbfgs"},       # Model 4
    {"C": 100.0, "solver": "saga"},       # Model 5
]

# --- Khởi tạo biến lưu mô hình tốt nhất ---
best_accuracy = 0
best_params = None
best_model = None

# (Tùy chọn) Nếu có server MLflow riêng, thiết lập Tracking URI ở đây
# mlflow.set_tracking_uri("http://localhost:5000")

# Khai báo 1 thí nghiệm tên là Customer_Classification_Experiment
mlflow.set_experiment("Customer_Classification_Experiment")

# --- Vòng lặp huấn luyện và đánh giá nhiều bộ tham số ---
for params in param_grid:
    with mlflow.start_run():  # Mỗi lần chạy sẽ tạo ra 1 run mới trong MLflow
        # Khởi tạo mô hình LogisticRegression với tham số tương ứng
        model = LogisticRegression(C=params["C"], solver=params["solver"], max_iter=1000)
        # Huấn luyện mô hình
        model.fit(X_train, y_train)
        # Dự đoán trên tập test
        y_pred = model.predict(X_test)
        # Tính độ chính xác
        acc = accuracy_score(y_test, y_pred)

        # Ghi lại tham số và kết quả vào MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")  # Lưu mô hình

        # In ra độ chính xác của từng mô hình
        print(f"Thử nghiệm: C={params['C']}, solver={params['solver']}, Accuracy={acc:.4f}")

        # Cập nhật mô hình tốt nhất nếu độ chính xác cao hơn
        if acc > best_accuracy:
            best_accuracy = acc
            best_params = params
            best_model = model

# --- Ghi lại mô hình tốt nhất và đăng ký vào Model Registry ---
if best_model is not None:
    with mlflow.start_run(run_name="Best_Model") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("best_accuracy", best_accuracy)
        model_info = mlflow.sklearn.log_model(best_model, artifact_path="best_model")
        model_uri = model_info.model_uri

        # Đăng ký mô hình với tên One_Customer_Classifier
        mlflow.register_model(
            model_uri=model_uri,
            name="One_Customer_Classifier"
        )

print("✅ Đã đăng ký mô hình One_Customer_Classifier vào Model Registry.")