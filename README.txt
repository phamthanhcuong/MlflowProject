Project: MLflow + Flask - Customer Classification Example


How to run:
- Install requirements: pip install -r requirements.txt
- Train models: python train.py
- mlflow ui hoặc mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000
- Start Flask app: python app.py


Kill hết PID chiếm 5000
Check PID
 lsof -i :5000
KILL
 kill -9 3686 3687 3688 3689 3690
Check lại
 lsof -i :5000
python app.py


CLean FLFlow
 rm -rf mlruns/
 rm -f mlflow.db
 bash reset_mlflow.sh
 mlflow ui

 Promote Model
 Tại chio tiết của 1 version Model
1: Tắt New model registry UI
2: Reload trang
3: Click  Stage : None chuyển thành Production

4: Confirm Transition Stage
