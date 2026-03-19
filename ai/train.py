import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# 1. 데이터 로드
df = pd.read_csv('data/patient_dataset.csv')

print("원본 데이터 크기:", df.shape)

# 2. 데이터 정제
df = df[df['visit_count'] >= 2]
df = df[df['gap_days'].notnull()]

print("정제 후 데이터 크기:", df.shape)

# 3. feature / label
features = ['gap_days', 'visit_count', 'days_since_visit']
X = df[features]
y = df['label_revisit_90d']

# 4. 결측값 처리
X = X.fillna(0)

# 5. train/test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Pipeline (스케일링 + 모델)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# 7. 학습
pipeline.fit(X_train, y_train)

# 8. 평가
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("정확도:", acc)

# 9. 모델 저장 (핵심)
joblib.dump({
    "model": pipeline,
    "features": features
}, 'model.pkl')

print("모델 저장 완료 (model.pkl)")