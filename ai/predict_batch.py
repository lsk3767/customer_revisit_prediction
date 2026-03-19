import sys
import json
import pandas as pd
import joblib
import os

try:
    # 모델 로드
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'model.pkl')

    saved = joblib.load(model_path)
    model = saved["model"]
    features = saved["features"]

    # 전체 데이터 입력
    input_data = sys.stdin.read()
    patients = json.loads(input_data)

    df = pd.DataFrame(patients)

    # NaN 처리
    df = df.fillna(0)

    # 🔥 첫 방문 환자 분리
    df['revisit_prob'] = 0.8  # 기본값

    mask = df['visit_count'] > 1

    if mask.sum() > 0:
        probs = model.predict_proba(df.loc[mask, features])[:, 1]
        df.loc[mask, 'revisit_prob'] = probs

    # 결과 반환
    result = df[['CUST_NO', 'revisit_prob']].to_dict(orient='records')

    print(json.dumps(result))

except Exception as e:
    print(f"ERROR: {str(e)}", file=sys.stderr)
    sys.exit(1)