import pandas as pd
import joblib
import sys
import json
import os

try:
    # 입력 받기
    input_json = sys.argv[1]
    data = json.loads(input_json)

    # 🔥 모델 경로 수정 (핵심)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'model.pkl')

    saved = joblib.load(model_path)
    model = saved["model"]
    features = saved["features"]

    # 데이터 변환
    X = pd.DataFrame([[
        data['gap_days'],
        data['visit_count'],
        data['days_since_visit']
    ]], columns=features)

    # 예측
    prob = model.predict_proba(X)[0][1]

    # 결과 출력
    print(json.dumps({
        "CUST_NO": data['CUST_NO'],
        "revisit_prob": float(prob)
    }))

except Exception as e:
    # 🔥 에러 로그 출력 (Node에서 확인 가능)
    print(f"ERROR: {str(e)}", file=sys.stderr)
    sys.exit(1)