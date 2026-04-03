import sys
import json
import pandas as pd
import joblib
import os

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'model_latest_2.pkl')

    saved = joblib.load(model_path)
    model = saved["model"]
    features = saved["features"]

    input_data = sys.stdin.read()
    patients = json.loads(input_data)

    df = pd.DataFrame(patients)

    # 타입 변환
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.fillna(0)

    # 예측
    probs = model.predict_proba(df[features])[:, 1]
    df['revisit_prob'] = probs

    result = df[['CUST_NO', 'revisit_prob']].to_dict(orient='records')

    # 이것만 stdout (중요)
    print(json.dumps(result))

except Exception as e:
    print(f"ERROR: {str(e)}", file=sys.stderr)
    sys.exit(1)