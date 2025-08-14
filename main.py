import streamlit as st
import pandas as pd
from modules import eda, anomaly_model, llm_selector, chatbot
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

st.set_page_config(page_title="Data Analysis AI Agent", layout="wide")

st.title("🖥️ Data Analysis AI Agent")

# 1. 데이터 업로드
uploaded_file = st.file_uploader("📁 데이터 파일을 업로드하세요 (CSV, Excel)", type=["csv", "xlsx"])

import os

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    df = None

    if file_ext == ".csv":
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")

        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="cp949")

        except pd.errors.EmptyDataError:
            st.error("⚠️ CSV 파일에 데이터가 없습니다.")
            df = None

        if df is not None:
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
                df_raw = pd.read_excel(xls, sheet_name=selected_sheet, engine="openpyxl", header=None)
                df = df_raw.iloc[start_row:]
                df = df.dropna(how='all')  # 전부 NaN인 행 제거
                df.columns = [str(col) if pd.notna(col) else f"Unnamed_{i}" for i, col in enumerate(df.iloc[0])]
                df = df[1:]  # 컬럼으로 쓴 행 제거
                st.write("📊 데이터 불러오기 완료")
                st.dataframe(df)
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
    st.subheader("LLM Chatbot")
    providers = chatbot.get_available_llms()
    if not providers:
        st.info("등록된 API 키가 없습니다. .env에 GOOGLE_API_KEY 또는 OPENAI_API_KEY를 설정하세요.")
    else:
        labels = [p[0] for p in providers]
        values = [p[1] for p in providers]
        sel_idx = st.selectbox("모델을 선택하세요", list(range(len(labels))), format_func=lambda i: labels[i])

        if "llm_ready" not in st.session_state:
            st.session_state.llm_ready = False

        if st.button("모델 활성화", key="btn_activate_llm"):
            init_msg = chatbot.init_llm(values[sel_idx])
            st.session_state.llm_ready = (init_msg == "ok")
            if not st.session_state.llm_ready:
                st.error(f"모델 초기화 실패: {init_msg}")
            else:
                st.success("모델이 활성화되었습니다. 이제 메시지를 입력해보세요!")

        # 상태 표시 + Ping 배지
        status = chatbot.get_status()
        if "ping_status" not in st.session_state:
            st.session_state["ping_status"] = None  # 'success' | 'fail' | None
            st.session_state["ping_msg"] = ""
            st.session_state["ping_time"] = None

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"상태: {'✅ Ready' if st.session_state.llm_ready else '⛔ Not Ready'}")
        with col2:
            st.caption(f"모델: {status.get('model') or '-'}")
        with col3:
            ping = st.session_state.get("ping_status")
            if ping == "success":
                st.caption(f"Ping: ✅ 성공 ({st.session_state.get('ping_time') or '-'})")
            elif ping == "fail":
                st.caption(f"Ping: ❌ 실패 ({st.session_state.get('ping_time') or '-'})")
            else:
                st.caption("Ping: -")

        if status.get('error') and not st.session_state.llm_ready:
            st.error(f"초기화 오류: {status['error']}")

        # 연결 테스트 (항상 표시, 준비 안 되면 버튼 비활성화)
        with st.expander("🔧 LLM 연결 테스트", expanded=False):
            ping_disabled = not st.session_state.llm_ready

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Ping LLM", key="btn_ping_llm", disabled=ping_disabled):
                    with st.spinner("핑 테스트 중..."):
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        logs_dir = os.path.join(os.getcwd(), "logs")
                        os.makedirs(logs_dir, exist_ok=True)
                        log_path = os.path.join(logs_dir, "llm_ping.log")

                        def _do_ping():
                            return chatbot.llm.invoke("한국어로 'pong' 한 단어만 출력해")

                        try:
                            with ThreadPoolExecutor(max_workers=1) as ex:
                                fut = ex.submit(_do_ping)
                                resp = fut.result(timeout=10)  # 10초 타임아웃
                            content = getattr(resp, 'content', str(resp))
                            st.session_state['ping_status'] = 'success'
                            st.session_state['ping_msg'] = content
                            st.session_state['ping_time'] = ts
                            st.success(f"LLM 응답: {content}")
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(f"[{ts}] SUCCESS: {content}\n")
                        except FutTimeout:
                            st.session_state['ping_status'] = 'fail'
                            st.session_state['ping_msg'] = '시간 초과(10초)'
                            st.session_state['ping_time'] = ts
                            st.error("테스트 실패: 시간 초과(10초)")
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(f"[{ts}] FAIL: Timeout 10s\n")
                        except Exception as e:
                            st.session_state['ping_status'] = 'fail'
                            st.session_state['ping_msg'] = str(e)
                            st.session_state['ping_time'] = ts
                            st.error(f"테스트 실패: {e}")
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(f"[{ts}] FAIL: {e}\n")
            with c2:
                if st.button("Ping 리셋", key="btn_ping_reset"):
                    st.session_state['ping_status'] = None
                    st.session_state['ping_msg'] = ''
                    st.session_state['ping_time'] = None
                    st.info("Ping 상태를 초기화했습니다.")

            # 마지막 Ping 결과 표시
            last = st.session_state.get('ping_status')
            if last:
                st.write(
                    f"마지막 결과: **{'성공' if last=='success' else '실패'}** @ {st.session_state.get('ping_time')} — {st.session_state.get('ping_msg')}"
                )

        # -- Chat-based interface using st.chat_input and st.chat_message --
        if st.session_state.llm_ready:
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            chat_history = st.session_state["chat_history"]

            use_tools = st.checkbox("파이썬 도구 사용(코드 실행)", value=False)

            # Render chat history
            for msg in chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            # Chat input
            user_input = st.chat_input("질문을 입력하세요")
            if user_input:
                # Add user message
                chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)
                # Get assistant reply
                reply = chatbot.ask_with_llm(user_input, df)
                chat_history.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.write(reply)