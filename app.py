# ---- APP.PY ----

# Import thư viện Flask (làm web server nhỏ)
from flask import Flask, request, jsonify
# Import MLflow để load mô hình đã lưu
import mlflow.pyfunc
# Import pandas để xử lý dữ liệu
import pandas as pd

# 1. Khởi tạo Flask App
app = Flask(__name__)

# 2. Load mô hình từ MLflow Model Registry
try:
    # Tải mô hình One_Customer_Classifier đã được promote thành Production
    model = mlflow.pyfunc.load_model(model_uri="models:/One_Customer_Classifier/Production")
    print("✅ Mô hình One_Customer_Classifier đã load thành công từ MLflow Model Registry.")
except Exception as e:
    # Nếu lỗi trong lúc load model, in ra lỗi
    print(f"❌ Lỗi khi tải mô hình: {e}")
    model = None

# 3. Tạo đường dẫn API để nhận yêu cầu dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        # Nếu mô hình chưa load được, trả về lỗi
        return jsonify({"error": "Mô hình chưa được load thành công."}), 500

    try:
        # Lấy dữ liệu JSON từ yêu cầu của người dùng
        data = request.get_json()
        if not data:
            return jsonify({"error": "Không nhận được dữ liệu đầu vào."}), 400

        # Chuyển dữ liệu JSON thành bảng pandas DataFrame
        input_df = pd.DataFrame([data])

        # Dự đoán kết quả bằng mô hình
        prediction = model.predict(input_df)

        # Trả kết quả dự đoán cho người dùng
        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as ex:
        # Nếu dự đoán lỗi, trả về lỗi
        return jsonify({"error": str(ex)}), 500

# 4. Chạy Flask app
if __name__ == '__main__':
    # Mở server web tại 0.0.0.0 (mọi địa chỉ), port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)