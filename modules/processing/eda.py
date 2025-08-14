# modules/eda.py
import streamlit as st

def run_eda(df):
    if df is None or df.empty:
        st.warning("데이터가 비어있거나 로딩되지 않았습니다.")
        return

    st.subheader("📊 기본 통계")
    st.write(df.describe())

    st.subheader("🧩 결측치")
    st.write(df.isnull().sum())

    st.subheader("🧪 컬럼 타입")
    st.write(df.dtypes)

def run_univariate_eda(df):
    st.subheader("📈 일반형 EDA (단변량 분석)")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if not numeric_cols.any():
        st.warning("수치형 컬럼이 존재하지 않습니다.")
        return

    selected_col = st.selectbox("분석할 수치형 컬럼을 선택하세요", numeric_cols)

    if selected_col:
        st.write(f"선택한 컬럼: `{selected_col}`")
        st.write("기초 통계:")
        st.write(df[selected_col].describe())
        st.line_chart(df[selected_col])


def run_multivariate_eda(df):
    st.subheader("📊 다변형 EDA (상관관계 분석)")
    numeric_cols = df.select_dtypes(include=["int64", "float64"])

    if numeric_cols.shape[1] < 2:
        st.warning("두 개 이상의 수치형 컬럼이 필요합니다.")
        return

    st.write("📌 수치형 컬럼 간 상관관계 분석")
    corr = numeric_cols.corr()
    st.dataframe(corr)
    st.write("✅ 상관계수 히트맵 ")

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams['font.family'] = 'AppleGothic'  # 한글 폰트 설정

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    except ImportError:
        st.error("시각화를 위해 seaborn과 matplotlib이 필요합니다. 설치 후 다시 시도해주세요.")