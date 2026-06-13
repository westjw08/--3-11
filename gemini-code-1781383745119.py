import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. 원본 실험 데이터셋 구축 (훈련 데이터)
train_data = {
    'Drink': ['Monster', 'The_King', 'Trevi', 'Pocari_Sweat', 'Gatorade'],
    'Caffeine': [100, 100, 0, 0, 0],
    'Sugar': [41, 40, 0, 30, 39],
    'pH': [3.4, 3.3, 4.5, 3.7, 3.0],
    'Corrosion': [8.09, 5.60, 4.84, 4.00, 3.87],
    'Denaturation_Time': [12, 15, 27, 22, 19]
}
df_train = pd.DataFrame(train_data)

# 2. 머신러닝 모델 학습을 위한 X(독립변수), Y(종속변수) 분리
# 입력 피처: 시중 제품 정보 (카페인, 당류, pH)
X_train = df_train[['Caffeine', 'Sugar', 'pH']]
# 출력 예측 대상 피처: 실험 결과치 (치아부식률, 단백질변성시간)
Y_train = df_train[['Corrosion', 'Denaturation_Time']]

# 3. 다중 선형 회귀(Multiple Linear Regression) 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, Y_train) # 학습 가중치와 편향을 스스로 연산하는 과정

# 4. 사용자 인터랙티브 예측 루프 구동
print("=== 🤖 머신러닝(ML) 기반 음료 유해성 예측 및 진단 시스템 ===")
print("현재 학습된 음료 데이터셋: Monster, The_King, Trevi, Pocari_Sweat, Gatorade")

while True:
    print("\n---------------------------------------------------")
    drink_name = input("▶ 진단할 새로운 음료의 이름을 입력하세요 (종료: 'q'): ").strip()
    if drink_name.lower() == 'q':
        break
        
    # 새로운 음료수 이름이 기존 데이터셋에 있는지 확인
    drink_list_lower = [d.lower() for d in df_train['Drink']]
    
    if drink_name.lower() in drink_list_lower:
        # 기존에 있는 음료면 원본 데이터 그대로 추출
        idx = drink_list_lower.index(drink_name.lower())
        caff = df_train.iloc[idx]['Caffeine']
        sugar = df_train.iloc[idx]['Sugar']
        ph = df_train.iloc[idx]['pH']
        corr = df_train.iloc[idx]['Corrosion']
        denat = df_train.iloc[idx]['Denaturation_Time']
        is_predicted = False
    else:
        # 🌟 데이터셋에 없는 새로운 음료라면 성분표 정보만 입력받음
        print(f"💡 '{drink_name}'은(는) 새로운 음료입니다. 제품 성분표 정보를 바탕으로 실험값을 예측합니다.")
        caff = float(input("1. 카페인 함량(mg)을 입력하세요: "))
        sugar = float(input("2. 당류 함량(g)을 입력하세요: "))
        ph = float(input("3. 음료의 pH를 입력하세요: "))
        
        # 머신러닝 모델을 통해 [부식률, 변성시간]을 추정(Prediction)
        X_new = pd.DataFrame([[caff, sugar, ph]], columns=['Caffeine', 'Sugar', 'pH'])
        Y_pred = model.predict(X_new)
        
        corr = Y_pred[0][0]
        denat = Y_pred[0][1]
        is_predicted = True
        
        # 만약 예측된 변성 시간이 음수가 나오거나 비현실적일 경우 최소 안전장치 예외 처리
        if denat <= 0: denat = 1.0 
        if corr < 0: corr = 0.0

    # 5. 실시간 데이터 융합 및 위험도 계산을 위한 임시 임베딩 데이터프레임 생성
    # 새로운 음료의 데이터가 포함된 전체 데이터 풀에서 정규화를 다시 진행해야 공정함
    temp_row = {'Drink': drink_name, 'Caffeine': caff, 'Sugar': sugar, 'pH': ph, 'Corrosion': corr, 'Denaturation_Time': denat}
    df_eval = pd.concat([df_train, pd.DataFrame([temp_row])], ignore_index=True)
    
    # 데이터 전처리 (산도 위험도 변환 및 변성 속도 변환)
    df_eval['pH_Risk'] = 7.0 - df_eval['pH']
    df_eval['Denaturation_Velocity'] = 1.0 / df_eval['Denaturation_Time']
    
    # Min-Max 정규화 수행
    features_to_norm = ['Caffeine', 'Sugar', 'pH_Risk', 'Corrosion', 'Denaturation_Velocity']
    for col in features_to_norm:
        df_eval[f'{col}_norm'] = (df_eval[col] - df_eval[col].min()) / (df_eval[col].max() - df_eval[col].min() + 1e-5)
        
    # 사용자 프로필 가중치 설정 (여기선 일반 모드 기준 예시, [Caff, Sugar, pH, Corr, Denat_Vel])
    weights = np.array([0.40, 0.15, 0.15, 0.15, 0.15])
    
    # 마지막 행(방금 입력/예측한 음료)의 정규화 벡터 추출 후 위험도 산출
    new_drink_idx = len(df_eval) - 1
    drink_vector = df_eval.iloc[new_drink_idx][[f'{c}_norm' for c in features_to_norm]].to_numpy()
    risk_score = np.dot(drink_vector, weights) * 100
    
    # 최종 결과 리포트
    print(f"\n[📊 {drink_name} 유해성 진단 결과 리포트]")
    if is_predicted:
        print(f"🤖 AI 추정 달걀껍데기 부식률: {corr:.2f}%")
        print(f"🤖 AI 추정 단백질 변성 시간: {denat:.2f}분")
    else:
        print(f"🔬 실제 실험 데이터 사용")
    print(f"🚨 종합 맞춤형 위험도 지수: {risk_score:.2f} / 100점")