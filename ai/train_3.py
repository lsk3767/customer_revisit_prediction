import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib
import os

# 1. 데이터 로드
df = pd.read_csv('data/patient_dataset_last.csv', encoding='utf-8')

print("원본 데이터 크기:", df.shape)

# 2. 데이터 정제
df = df[df['방문횟수'] >= 2]
df = df[df['방문간격'].notnull()]

print("정제 후 데이터 크기:", df.shape)

# 컬럼명 변환
df = df.rename(columns={
    '방문간격': 'gap_days',
    '방문횟수': 'visit_count',
    '평균방문간격': 'avg_gap_days',
    '최근방문여부': 'recent_visit',
    '장기미방문여부': 'long_term_no_visit',
    '이전방문간격': 'prev_gap',
    '방문간격변화량': 'gap_change',
    '방문지연여부': 'delay_flag',
    '최근3회평균간격': 'recent3_avg',
    '방문간격비율': 'gap_ratio',
    '재방문여부': 'target'
})

#  중요: 방문일 컬럼 확인 
VISIT_DATE_COL = '방문일자'  # ← 실제 컬럼명으로 수정 필요

# 3. feature / label
features = [
    'gap_days',
    'visit_count',
    'avg_gap_days',
    'recent_visit',
    'long_term_no_visit',
    'prev_gap',
    'gap_change',
    'delay_flag',
    'recent3_avg',
    'gap_ratio'
]

X = df[features]
y = df['target']

print("===== LABEL 분포 =====")
print(y.value_counts())

# 4. 결측값 처리
X = X.fillna(0)

# 5. 환자 단위 split
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['고객번호']))

# train
X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

# test (원본 df 기준으로 가져오기)
test_df = df.iloc[test_idx].copy()

# 핵심: 환자별 마지막 방문만 추출
test_df = test_df.groupby('고객번호').filter(lambda x: len(x) >= 2)

test_last = test_df.sort_values(['고객번호', VISIT_DATE_COL]) \
                  .groupby('고객번호').nth(-2)

print("\n===== TEST 환자 수 =====")
print(test_last['고객번호'].nunique())

# 다시 feature / label 분리
X_test = test_last[features].fillna(0)
y_test = test_last['target']

# 6. 모델 정의
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y[y==0]) / len(y[y==1]),
    random_state=42,
    n_jobs=-1
)

# 7. 학습
model.fit(X_train, y_train)

# 8. 평가
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\n===== 확률 분포 =====")
print("MIN:", y_prob.min())
print("MAX:", y_prob.max())
print("SAMPLE:", y_prob[:10])

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n=== 모델 성능 ===")
print("정확도:", acc)
print("AUC:", auc)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# 9. Feature 중요도
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)

# 10. 모델 저장
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model_latest_2.pkl')

joblib.dump({
    "model": model,
    "features": features
}, model_path)

print(f"\n모델 저장 완료: {model_path}")