import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ─── 1. 웹 페이지 기본 설정 및 디자인 ───
st.set_page_config(
    page_title="청소년 음료 유해성 진단 AI",
    page_icon="🥤",
    layout="centered"
)

# 이화여대 컬러 감성을 담은 부드러운 그린/크린조 톤의 메인 타이틀 디자인
st.title("🥤 청소년 맞춤형 음료 유해성 진단 AI 시스템")
st.markdown("""
    본 시스템은 사용자가 입력한 음료 성분을 바탕으로, **머신러닝(다중 선형 회귀) 모델**이 생체 유해성 수치를 예측하고 
    사용자의 건강 프로필에 맞춰 **종합 위험도 지수**를 실시간으로 컴퓨팅합니다.
""")
st.markdown("---")

# ─── 2. 기본 실험 데이터셋 및 머신러닝 모델 초기화 (캐싱 처리) ───
@st.cache_data
def train_ml_model():
    # 보내준 수정본 데이터 반영 (트레비 27분, 포카리 22분, 게토레이 19분)
    train_data = {
        'Drink': ['Monster', 'The_King', 'Trevi', 'Pocari_Sweat', 'Gatorade'],
        'Caffeine': [100, 100, 0, 0, 0],
        'Sugar': [41, 40, 0, 30, 39],
        'pH': [3.4, 3.3, 4.5, 3.7, 3.0],
        'Corrosion': [8.09, 5.60, 4.84, 4.00, 3.87],
        'Denaturation_Time': [12, 15, 27, 22, 19]
    }
    df_train = pd.DataFrame(train_data)
    
    X_train = df_train[['Caffeine', 'Sugar', 'pH']]
    Y_train = df_train[['Corrosion', 'Denaturation_Time']]
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model, df_train

model, df_train = train_ml_model()

# ─── 3. 사이드바: 현재 학습된 데이터 풀 시각화 ───
st.sidebar.header("📊 시스템 학습 데이터 원본")
st.sidebar.write("네가 직접 수행한 5종의 아날로그 실험 데이터셋이야.")
st.sidebar.dataframe(df_train, use_container_width=True)

# ─── 4. 메인 화면: 유저 입력 인터페이스 구축 ───
st.header("👤 1단계: 유저 정보 및 음료 선택")

# 음료 선택 방식 (기존 음료 선택 혹은 새로운 음료 성분 직접 입력)
drink_option = st.selectbox(
    "진단할 음료 유형을 선택하세요:",
    ["기존 실험 데이터셋에서 선택", "새로운 제3의 음료 성분 입력 (AI 예측)"]
)

# 유저 상태 프로필 가중치 매핑
user_profile = st.radio(
    "현재 본인의 건강 취약 상태를 선택하세요:",
    ("일반 청소년 기준 모드", "위 건강 취약 모드 (속 쓰림, 위염 등)", "치아 건강 취약 모드 (충치, 교정 등)"),
    horizontal=True
)

st.markdown("### 🧪 2단계: 성분 데이터 입력")

if drink_option == "기존 실험 데이터셋에서 선택":
    selected_drink = st.selectbox("음료 이름을 골라주세요:", df_train['Drink'].tolist())
    idx = df_train[df_train['Drink'] == selected_drink].index[0]
    
    caff = float(df_train.iloc[idx]['Caffeine'])
    sugar = float(df_train.iloc[idx]['Sugar'])
    ph = float(df_train.iloc[idx]['pH'])
    corr = float(df_train.iloc[idx]['Corrosion'])
    denat = float(df_train.iloc[idx]['Denaturation_Time'])
    is_predicted = False
    
    # 선택된 기존 음료의 스펙을 화면에 가볍게 노출
    st.info(f"✨ 선택된 **{selected_drink}**의 성분 정보 -> 카페인: {caff}mg | 당류: {sugar}g | pH: {ph}")

else:
    # 새로운 음료 선택 시 슬라이더와 입력창 노출
    new_drink_name = st.text_input("새로운 음료의 이름을 입력하세요:", "핫식스")
    selected_drink = new_drink_name if new_drink_name else "Unknown Drink"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        caff = st.number_input("카페인 함량 (mg):", min_value=0.0, max_value=500.0, value=60.0, step=5.0)
    with col2:
        sugar = st.number_input("당류 함량 (g):", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
    with col3:
        ph = st.slider("음료의 산도 (pH):", min_value=0.0, max_value=14.0, value=3.0, step=0.1)
        
    # 새로운 성분을 기반으로 머신러닝 모델 구동 (predict)
    X_new = pd.DataFrame([[caff, sugar, ph]], columns=['Caffeine', 'Sugar', 'pH'])
    Y_pred = model.predict(X_new)
    
    corr = Y_pred[0][0]
    denat = Y_pred[0][1]
    is_predicted = True
    
    # 비현실적 예측값 보정용 안전장치
    if denat <= 0: denat = 1.0
    if corr < 0: corr = 0.0

# ─── 5. 위험도 연산 및 대시보드 출력 ───
st.markdown("---")
st.header("🚨 3단계: AI 실시간 유해성 진단 결과")

if st.button("🧬 종합 위험도 지수 산출하기", type="primary"):
    
    # 실시간 데이터 풀 정규화를 위한 가상 임베
