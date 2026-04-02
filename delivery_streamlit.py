"""
배달 소요 시간 예측 - Streamlit 웹 앱
실행: streamlit run delivery_streamlit.py
※ model.pkl, train.csv 가 같은 폴더에 있어야 합니다
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🛵 배달 소요 시간 예측",
    page_icon="🛵",
    layout="wide"
)


@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')

    df['Time_taken(min)'] = (
        df['Time_taken(min)'].astype(str)
        .str.replace('(min)', '', regex=False).str.strip().astype(float)
    )
    df['Delivery_person_Age']     = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
    df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
    df['multiple_deliveries']     = pd.to_numeric(
        df['multiple_deliveries'].astype(str).str.strip(), errors='coerce')
    df['Weatherconditions']       = df['Weatherconditions'].astype(str).str.strip().str.replace('conditions ', '', regex=False)
    df['Road_traffic_density']    = df['Road_traffic_density'].astype(str).str.strip()
    df['Type_of_vehicle']         = df['Type_of_vehicle'].astype(str).str.strip()
    df['Type_of_order']           = df['Type_of_order'].astype(str).str.strip()
    df['Festival']                = df['Festival'].astype(str).str.strip()
    df['City']                    = df['City'].astype(str).str.strip()

    df['distance_km'] = np.sqrt(
        (df['Delivery_location_latitude']  - df['Restaurant_latitude'])**2 +
        (df['Delivery_location_longitude'] - df['Restaurant_longitude'])**2
    ) * 111

    df['order_hour'] = df['Time_Orderd'].astype(str).str[:2].str.strip()
    df['order_hour'] = pd.to_numeric(df['order_hour'], errors='coerce')
    return df

# 모델 로드 시도
try:
    saved = load_model()
    model   = saved['model']
    scaler  = saved['scaler']
    le_dict = saved['le_dict']
    FEATURES = saved['features']
    metrics  = saved['metrics']
    model_loaded = True
except:
    model_loaded = False

df = load_data()


st.sidebar.title("🛵 배달 소요 시간 예측")
st.sidebar.markdown("---")

if model_loaded:
    st.sidebar.success("✅ 모델 로드 완료")
    st.sidebar.metric("R² Score", f"{metrics['R2']:.4f}")
    st.sidebar.metric("MAE",      f"{metrics['MAE']:.2f}분")
    st.sidebar.metric("RMSE",     f"{metrics['RMSE']:.2f}분")
else:
    st.sidebar.error("❌ model.pkl 없음\ndelivery_time_prediction.py 먼저 실행!")

st.sidebar.markdown("---")
menu = st.sidebar.radio("메뉴", ["📊 데이터 분석", "🤖 모델 성능", "🔮 배달 시간 예측"])


if menu == "📊 데이터 분석":
    st.title("📊 배달 데이터 탐색적 분석 (EDA)")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 데이터",     f"{len(df):,}건")
    c2.metric("평균 소요 시간", f"{df['Time_taken(min)'].mean():.1f}분")
    c3.metric("최소 소요 시간", f"{df['Time_taken(min)'].min():.0f}분")
    c4.metric("최대 소요 시간", f"{df['Time_taken(min)'].max():.0f}분")
    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("소요 시간 분포")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['Time_taken(min)'], bins=30, color='steelblue', edgecolor='white')
        ax.set_xlabel('Delivery Time (min)')
        ax.set_ylabel('Count')
        st.pyplot(fig); plt.close()

    with col_b:
        st.subheader("교통 상황별 평균 소요 시간")
        fig, ax = plt.subplots(figsize=(7, 4))
        order  = ['Low', 'Medium', 'High', 'Jam']
        means  = [df[df['Road_traffic_density']==t]['Time_taken(min)'].mean() for t in order]
        bars   = ax.bar(order, means, color=['green','orange','tomato','darkred'], edgecolor='white')
        for bar, val in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.1f}', ha='center')
        ax.set_ylabel('Avg Time (min)')
        st.pyplot(fig); plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("날씨별 평균 소요 시간")
        fig, ax = plt.subplots(figsize=(7, 4))
        w_means = df.groupby('Weatherconditions')['Time_taken(min)'].mean().sort_values()
        ax.barh(w_means.index, w_means.values, color='mediumpurple', edgecolor='white')
        ax.set_xlabel('Avg Time (min)')
        st.pyplot(fig); plt.close()

    with col_d:
        st.subheader("시간대별 평균 소요 시간")
        fig, ax = plt.subplots(figsize=(7, 4))
        h_means = df.groupby('order_hour')['Time_taken(min)'].mean().sort_index()
        ax.plot(h_means.index, h_means.values, color='royalblue', marker='o', markersize=4, lw=2)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Avg Time (min)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

    st.subheader("상관관계 히트맵")
    num_cols = ['Delivery_person_Age','Delivery_person_Ratings',
                'distance_km','Vehicle_condition','multiple_deliveries','Time_taken(min)']
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
                cmap='coolwarm', ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    st.pyplot(fig); plt.close()


elif menu == "🤖 모델 성능":
    st.title("🤖 모델 성능 평가")
    st.markdown("---")

    if not model_loaded:
        st.warning("model.pkl이 없습니다. delivery_time_prediction.py를 먼저 실행해주세요.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{metrics['R2']:.4f}")
        c2.metric("MAE",      f"{metrics['MAE']:.2f}분")
        c3.metric("RMSE",     f"{metrics['RMSE']:.2f}분")
        st.markdown("---")
        st.info("전체 모델 비교 그래프 및 특성 중요도는 delivery_time_prediction.py 실행 후 생성된 PNG 파일을 확인하세요.")

        # 저장된 그래프 파일 표시 시도
        import os
        for fname, title in [
            ('03_Model_Evaluation.png', '모델 성능 비교'),
            ('04_Feature_Importance.png', '특성 중요도'),
            ('05_Actual_vs_Predicted.png', '실제 vs 예측값'),
        ]:
            if os.path.exists(fname):
                st.subheader(title)
                st.image(fname)

elif menu == "🔮 배달 시간 예측":
    st.title("🔮 배달 소요 시간 예측")
    st.markdown("정보를 입력하면 예상 배달 시간을 알려드립니다.")
    st.markdown("---")

    if not model_loaded:
        st.warning("model.pkl이 없습니다. delivery_time_prediction.py를 먼저 실행해주세요.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📦 배달 정보")
            distance     = st.slider("배달 거리 (km)", 0.5, 20.0, 3.0, 0.5)
            traffic      = st.selectbox("교통 상황", ['Low', 'Medium', 'High', 'Jam'])
            weather      = st.selectbox("날씨", ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Stormy', 'Sandstorms'])
            multi_del    = st.selectbox("다중 배달 수", [0, 1, 2, 3])
            festival     = st.radio("축제 기간", ['No', 'Yes'], horizontal=True)
            order_type   = st.selectbox("주문 종류", ['Snack', 'Drinks', 'Buffet', 'Meal'])
            order_hour   = st.slider("주문 시간 (시)", 0, 23, 12)
            pickup_wait  = st.slider("픽업 대기 시간 (분)", 0, 60, 10)

        with col2:
            st.subheader("🛵 배달원 정보")
            age          = st.slider("배달원 나이", 18, 50, 29)
            rating       = st.slider("배달원 평점", 2.5, 5.0, 4.5, 0.1)
            vehicle      = st.selectbox("차량 종류", ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'])
            veh_cond     = st.selectbox("차량 상태 (0=좋음, 3=나쁨)", [0, 1, 2, 3])
            city         = st.selectbox("도시 유형", ['Urban', 'Semi-Urban', 'Metropolitian'])

        st.markdown("---")

        if st.button("🚀 배달 시간 예측하기", use_container_width=True):
            def safe_encode(le, val):
                try:    return le.transform([val])[0]
                except: return 0

            input_dict = {
                'Delivery_person_Age':     age,
                'Delivery_person_Ratings': rating,
                'distance_km':             distance,
                'Vehicle_condition':       veh_cond,
                'multiple_deliveries':     multi_del,
                'pickup_wait_min':         pickup_wait,
                'order_hour':              order_hour,
                'Weatherconditions':       safe_encode(le_dict['Weatherconditions'], weather),
                'Road_traffic_density':    safe_encode(le_dict['Road_traffic_density'], traffic),
                'Type_of_order':           safe_encode(le_dict['Type_of_order'], order_type),
                'Type_of_vehicle':         safe_encode(le_dict['Type_of_vehicle'], vehicle),
                'Festival':                safe_encode(le_dict['Festival'], festival),
                'City':                    safe_encode(le_dict['City'], city),
            }

            X_input = pd.DataFrame([input_dict])[FEATURES]
            X_scaled = scaler.transform(X_input)
            pred = model.predict(X_scaled)[0]

            st.success(f"### 🕐 예상 배달 시간: **{pred:.0f}분**")

            c1, c2, c3 = st.columns(3)
            c1.metric("예상 소요 시간", f"{pred:.0f}분")
            c2.metric("배달 거리",      f"{distance}km")
            c3.metric("교통 상황",      traffic)

            st.markdown("---")
            st.subheader("📌 주요 영향 요인 분석")
            traffic_map = {'Low':0,'Medium':5,'High':12,'Jam':25}
            weather_map = {'Sunny':0,'Cloudy':3,'Windy':5,'Fog':8,'Stormy':15,'Sandstorms':18}
            factors = {
                '거리':         round(distance * 3.5, 1),
                '교통 상황':    traffic_map.get(traffic, 0),
                '날씨':         weather_map.get(weather, 0),
                '다중 배달':    multi_del * 7,
                '픽업 대기':    pickup_wait,
                '축제 기간':    10 if festival == 'Yes' else 0,
                '배달원 평점':  round((5 - rating) * 2, 1),
            }
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(list(factors.keys()), list(factors.values()),
                    color='steelblue', edgecolor='white')
            ax.set_xlabel('Estimated Impact (min)')
            ax.set_title('Contribution by Factor')
            st.pyplot(fig); plt.close()