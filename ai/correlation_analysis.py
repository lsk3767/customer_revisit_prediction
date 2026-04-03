import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('data/patient_dataset_last.csv')

# 확인
print(df.head())

features = [
    '방문간격',
    '방문횟수',
    '평균방문간격',
    '최근방문여부',
    '장기미방문여부',
    '최근3회평균간격',
    '방문간격변화량',
    '방문지연여부',
    '이전방문간격'
   
]

target = '재방문여부'

df = df[features + [target]]
df = df.fillna(df.mean(numeric_only=True))

corr = df.corr()

print(corr)

target_corr = corr[target].sort_values(ascending=False)

print("\n=== Target Correlation ===")
print(target_corr)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

target_corr.drop(target).plot(kind='bar')
plt.title('Feature vs Target Correlation')
plt.xticks(rotation=45)
plt.show()
