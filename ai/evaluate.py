import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

# 데이터 로드
df = pd.read_csv('data/evaluate.csv')

X = df[['gap_days', 'visit_count', 'days_since_visit']]
y = df['target']

X = X.fillna(X.mean())

# 모델 로드
data = joblib.load('model_latest.pkl')

model = data['model']
scaler = data.get('scaler')

# 스케일 적용 (있으면)
if scaler:
    X = scaler.transform(X)

# 예측
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:,1]

print("=== Classification Report ===")
print(classification_report(y, y_pred))

print("=== AUC ===")
print(roc_auc_score(y, y_prob))