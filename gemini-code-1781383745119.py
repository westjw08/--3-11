# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ─── [웹 UI 설정] ───
st.set_page_config(
    page_title="음료 유해성 진단 AI 시스템",
    page_icon="🥤",
    layout="centered"
)

st.title("🥤 머신러닝 기반 음료 유해성 진단 시스템")
st.markdown("""
    이 웹사이트는 **다중 선형 회귀(Linear Regression)** AI 모델을 사용하여 
    실험하지 않은 새로운 음료의 유해성 수치를 예측하고, 맞춤형 위험 지수를 산출합니다.
""")
st.markdown("---")

# ─── [1단계: 머신러닝 모델 학습 (최초 1회만 실행 및 캐싱)] ───
@st.cache_data
def train_ml_model():
    # 원본 실험 데이터셋 구축 (보내주신 수치 반영)
    train_data = {
        'Drink': ['Monster', 'The_King', 'Trevi', 'Pocari_Sweat', 'Gatorade'],
        'Caffeine': [100.0, 100.0, 0.0, 0.0, 0.0],
        'Sugar': [41.0, 40.0, 0.0, 30.0, 39.0],
        'pH': [3.4, 3.3, 4.5, 3.7, 3.0],
        'Corrosion': [8.09, 5.60, 4.84, 4.00, 3.87],
        'Denaturation_Time': [12.0, 15.0, 27.0, 22.0, 19.0]
    }
    df_train = pd.DataFrame(train_data)
    
    # 독립변수(X)와 종속변수(Y) 분리
    X_train = df_train[['Caffeine', 'Sugar', 'pH']]
    Y_train = df_train[['Corrosion', 'Denaturation_Time']]
    
    # 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model, df_train

model, df_train = train_ml_model()

# ─── [사이드바: 최신 Streamlit 버전에 맞춤 변경] ───
st.sidebar.header("📊 AI 모델 훈련 데이터셋")
# 🌟 경고 메시지를 없애기 위해 use_container_width=True 대신 width="stretch" 사용
st.sidebar.dataframe(df_train, width="stretch")

# ─── [2단계: 웹 입력 인터페이스 구현] ───
st.header("🔍 음료 정보 입력")

drink_name = st.text_input("진단할 음료의 이름을 입력하세요:", "핫식스").strip()

# 대소문자 구분 없이 기존 데이터셋에 있는지 검사
drink_list_lower = [d.lower() for d in df_train['Drink']]

# 데이터 제어를 위한 세션 초기화
caff, sugar, ph, corr, denat = 0.0, 0.0, 0.0, 0.0, 0.0
is_predicted = False

if drink_name.lower() in drink_list_lower:
    # 1) 기존에 실험했던 음료인 경우
    st.success(f"🔬 '{drink_name}'은(는) 기존 데이터셋에 있는 음료입니다. 실제 실험 수치를 가져옵니다.")
    idx = drink_list_lower.index(drink_name.lower())
    
    caff = float(df_train.iloc[idx]['Caffeine'])
    sugar = float(df_train.iloc[idx]['Sugar'])
    ph = float(df_train.iloc[idx]['pH'])
    corr = float(df_train.iloc[idx]['Corrosion'])
    denat = float(df_train.iloc[idx]['Denaturation_Time'])
    is_predicted = False
    
    # 기존 성분 표시
    st.text(f"📋 고유 성분 정보 -> 카페인: {caff}mg | 당류: {sugar}g | pH: {ph}")

else:
    # 2) 데이터셋에 없는 새로운 음료인 경우 -> 성분표 직접 입력 유도
    st.info(f"💡 '{drink_name}'은(는) 새로운 음료입니다. 성분표 정보를 기반으로 실험값을 예측합니다.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        caff = st.number_input("1. 카페인 함량 (mg):", min_value=0.0, value=60.0, step=5.0)
    with col2:
        sugar = st.number_input("2. 당류 함량 (g):", min_value=0.0, value=30.0, step=1.0)
    with col3:
        ph = st.number_input("3. 음료의 pH (산도):", min_value=0.0, max_value=14.0, value=3.0, step=0.1)
        
    # 머신러닝 추정 실행 (오류 방지를 위해 컬럼명이 매칭된 DataFrame으로 예측)
    X_new = pd.DataFrame([[caff, sugar, ph]], columns=['Caffeine', 'Sugar', 'pH'])
    Y_pred = model.predict(X_new)
    
    corr = float(Y_pred[0][0])
    denat = float(Y_pred[0][1])
    is_predicted = True
    
    # 비현실적 데이터 방지 예외 처리 예외 보정
    if denat <= 0: denat = 1.0
    if corr < 0: corr = 0.0

# ─── [3단계: 위험도 연산 및 대시보드 출력] ───
st.markdown("---")
if st.button("🧬 실시간 유해성 진단 및 결과 산출", type="primary"):
    
    # 실시간 데이터 융합 및 정규화를 위한 데이터프레임 병합
    temp_row = {
        'Drink': drink_name, 
        'Caffeine': caff, 
        'Sugar': sugar, 
        'pH': ph, 
        'Corrosion': corr, 
        'Denaturation_Time': denat
    }
    df_eval = pd.concat([df_train, pd.DataFrame([temp_row])], ignore_index=True)
    
    # 전처리 엔지니어링 수행
    df_eval['pH_Risk'] = 7.0 - df_eval['pH']
    df_eval['Denaturation_Velocity'] = 1.0 / df_eval['Denaturation_Time']
    
    # Min-Max 정규화 파이프라인
    features_to_norm = ['Caffeine', 'Sugar', 'pH_Risk', 'Corrosion', 'Denaturation_Velocity']
    for col in features_to_norm:
        min_val = df_eval[col].min()
        max_val = df_eval[col].max()
        if max_val - min_val == 0:
            df_eval[f'{col}_norm'] = 0.0
        else:
            df_eval[f'{col}_norm'] = (df_eval[col] - min_val) / (max_val - min_val + 1e-5)
            
    # 가중치 결합 (일반 모드 기준 벡터 정의)
    weights = np.array([0.40, 0.15, 0.15, 0.15, 0.15])
    
    # 최하단에 위치한 현재 검사 대상 음료의 벡터 행 추출
    new_drink_idx = len(df_eval) - 1
    drink_vector = df_eval.iloc[new_drink_idx][[f'{c}_norm' for c in features_to_norm]].to_numpy()
    
    # 행렬 곱 연산을 통한 종합 위험도 점수 도출
    risk_score = np.dot(drink_vector, weights) * 100
    
    # 웹 화면 시각화 리포트 구성
    st.subheader(f"📊 [{drink_name}] 유해성 진단 리포트")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric(label="🚨 종합 위험도 지수", value=f"{risk_score:.2f} / 100")
    with col_res2:
        st.metric(label="🦷 AI 추정 치아 부식률", value=f"{corr:.2f} %")
    with col_res3:
        st.metric(label="🍳 AI 추정 단백질 변성 시간", value=f"{denat:.2f} 분")
        
    # 결과 해석 코멘트 분기
    if risk_score >= 70:
        st.error("⚠️ [위험] 이 음료는 현재 입력된 성분 조합상 청소년기 신체 건강에 매우 유해한 지표를 보입니다. 섭취 빈도를 크게 줄여야 합니다.")
    elif risk_score >= 40:
        st.warning("⚠️ [주의] 과다 복용 시 신체 손상 우려가 존재합니다. 공복 상태에서의 음용을 지양하세요.")
    else:
        st.success("🍏 [보통] 상대적으로 유해성이 덜한 지표를 나타냅니다. 정량 이내의 안전한 음용을 권장합니다.")
        
    # 하단 데이터 출처 가이드라인 명시
    if is_predicted:
        st.caption("🤖 *Notice: 치아 부식률 and 단백질 변성 시간은 다중 선형 회귀 AI 모델이 기존 실험의 규칙성을 바탕으로 학습 및 보간해 낸 예측치입니다.*")
    else:
        st.caption("🔬 *Notice: 이 리포트는 과학실에서 정밀하게 도출된 실제 물리적 실험 데이터 레이블을 기반으로 작동하였습니다.*")
