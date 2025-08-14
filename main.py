import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import chardet
from modules import eda, anomaly_model
from modules.llm_selector import LLMManager
from modules.chatbot import ask_with_llm
from modules.chatbot_conversational_memory import create_memory_conversational_chain, extract_analysis_intent_and_column
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout


st.set_page_config(page_title="Data Analysis AI Agent", layout="wide")

st.title("🖥️ Data Analysis AI Agent")

llm_manager = LLMManager()

def read_file(file):
    if file.name.endswith('.csv'):
        rawdata = file.read()
        result = chardet.detect(rawdata)
        encoding = result['encoding']
        file.seek(0)  # Move the file pointer back to the beginning
        return pd.read_csv(file, encoding=encoding)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)

# 1. 데이터 업로드
uploaded_file = st.file_uploader("📁 데이터 파일을 업로드하세요 (CSV, Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df = read_file(uploaded_file)

    if df is None:
        st.stop()

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if file_ext == ".csv":
        st.write("📊 데이터 불러오기 완료")
        st.dataframe(df)

    elif file_ext == ".xlsx":
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            selected_sheet = st.selectbox("📑 시트를 선택하세요", sheet_names)
            df_preview = pd.read_excel(xls, sheet_name=selected_sheet, engine="openpyxl", nrows=10)
            st.write("🔎 시트 미리보기 (상위 10행)")
            st.dataframe(df_preview)

            start_row = st.number_input("데이터가 시작되는 행 번호를 입력하세요 (0부터 시작)", min_value=0, value=0, step=1)

            if "load_clicked" not in st.session_state:
                st.session_state["load_clicked"] = False

            if st.button("이 시트를 불러와서 분석하기"):
                st.session_state["load_clicked"] = True

            if st.session_state["load_clicked"]:
                try:
                    df_raw = pd.read_excel(xls, sheet_name=selected_sheet, engine="openpyxl", header=None)
                    df = df_raw.iloc[start_row:]
                    df = df.dropna(how='all')  # 전부 NaN인 행 제거
                    df.columns = [str(col) if pd.notna(col) else f"Unnamed_{i}" for i, col in enumerate(df.iloc[0])]
                    df = df[1:]  # 컬럼으로 쓴 행 제거
                    st.write("📊 데이터 불러오기 완료")
                    st.dataframe(df)
                except Exception as e:
                    st.warning("Preprocessing failed: " + str(e))
                    df = None
        except Exception as e:
            st.error(f"Excel 파일 읽기 실패: {e}")
            df = None


    # 2. EDA 분석
    if df is not None:
        eda_type = st.radio("📌 EDA 유형을 선택하세요", ("일반형 EDA", "다변형 EDA"))


        if "run_eda_clicked" not in st.session_state:
            st.session_state["run_eda_clicked"] = False

        if st.button("EDA 분석 실행"):
            st.session_state["run_eda_clicked"] = True

        if st.session_state["run_eda_clicked"]:
            if eda_type == "일반형 EDA":
                eda.run_univariate_eda(df)
            elif eda_type == "다변형 EDA":
                eda.run_multivariate_eda(df)

    # 3. 이상탐지
    st.subheader("이상탐지 모델")
    selected_model = st.selectbox("모델을 선택하세요", ["Logistic Regression", "XGBoost", "AutoEncoder"])
            
    if st.button("이상탐지 실행"):
        result_df = anomaly_model.run_anomaly_detection(df, selected_model)
        st.dataframe(result_df)

    # 4. LLM Chatbot
    def show_chatbot_section(df):
        st.markdown("## LLM Chatbot")

        selected_provider = st.selectbox("🧠 사용할 LLM 모델을 선택하세요", ["google:gemini-1.5-flash", "openai:gpt-3.5-turbo"])

        if st.button("LLM 초기화"):
            result = llm_manager.init(selected_provider)
            if result == "ok":
                st.success(f"✅ LLM 초기화 완료 ({selected_provider})")
            else:
                st.error(f"❌ LLM 초기화 실패: {llm_manager.get_error()}")

        if "llm" not in st.session_state or selected_provider != st.session_state.get("last_loaded_model"):
            result = llm_manager.init(selected_provider)
            if result != "ok":
                st.error(f"LLM 초기화 실패: {llm_manager.get_error()}")
                st.stop()
            st.session_state["llm"] = llm_manager.get_llm()
            st.session_state["last_loaded_model"] = selected_provider

        llm = st.session_state.get("llm")
        if llm:
            st.success("✅ LLM 모델 로드 완료")

        import streamlit.components.v1 as components

        # 커스텀 CSS와 JS를 적용한 채팅 입력창
        st.markdown("""
        <style>
        .chat-input-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .chat-input-box {
            flex-grow: 1;
        }
        .chat-message.user {
            text-align: right;
            color: #00BFFF;
            margin-bottom: 1em;
        }
        .chat-message.bot {
            text-align: left;
            color: #00FF7F;
            margin-bottom: 1em;
        }
        </style>
        <script>
        const inputBox = window.parent.document.querySelector('input[data-testid="stTextInput"]');
        if (inputBox) {
          inputBox.addEventListener("keydown", function(event) {
            if (event.key === "Enter") {
              const button = window.parent.document.querySelector('button[kind="secondary"]');
              if (button) button.click();
            }
          });
        }
        </script>
        """, unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("질문을 입력하세요", key="chat_input_form")
            submitted = st.form_submit_button("질문하기")

        if submitted and user_input and 'df' in locals() and df is not None:
            try:
                df_preview = df.head(10).round(2).astype(str).to_string(index=False)
                chain = create_memory_conversational_chain(llm, df_preview)
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: chain.invoke({
                        "df_preview": df_preview,
                        "history": "\n".join([]),
                        "input": user_input
                    }, config={"configurable": {"session_id": "default"}}))
                    try:
                        answer = future.result(timeout=15)
                        st.write("✅ 응답:", answer)
                    except FutTimeout:
                        st.error("⏰ LLM 응답이 너무 오래 걸립니다. 프롬프트나 질문을 바꿔보세요.")
            except Exception as e:
                import traceback
                st.error(f"❌ 에이전트 실행 중 오류 발생: {e}")
                print(traceback.format_exc())
        else:
            st.warning("데이터를 먼저 업로드하거나 질문을 입력해주세요.")

        st.markdown("### 💬 이전 대화")
        from langchain_community.chat_message_histories import StreamlitChatMessageHistory
        msg_history = StreamlitChatMessageHistory()
        for msg in msg_history.messages:
            if msg.type == "human":
                st.markdown(f"<div class='chat-message user'>👤 {msg.content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message bot'>🤖 {msg.content}</div>", unsafe_allow_html=True)

        with st.expander("📂 Export Chat Logs"):
            if os.path.exists("logs/chat_log.csv"):
                with open("logs/chat_log.csv", "rb") as f:
                    st.download_button(
                        label="📥 Download chat_log.csv",
                        data=f,
                        file_name="chat_log.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No chat logs available yet.")

    show_chatbot_section(df)