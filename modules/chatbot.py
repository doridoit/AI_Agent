from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

def ask_with_llm(llm: BaseChatModel, user_input: str, df: pd.DataFrame, chat_history: list[str] = []) -> str:
    """
    사용자의 질문과 데이터프레임을 기반으로 LLM에게 질문하고 응답을 반환하는 체이닝 기반 함수
    """
    # 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 Python 기반 데이터 분석 AI야. 사용자의 질문에 정확하게 답변해야 해."),
        ("human", "다음은 데이터의 일부 미리보기야:\n{df_preview}"),
        ("human", "이전 대화 기록은 다음과 같아:\n{history}"),
        ("human", "질문: {input}")
    ])

    # 출력 파서 (텍스트만 출력)
    parser = StrOutputParser()

    # 체이닝 정의
    chain: RunnableSequence = prompt | llm | parser

    # 실제 실행
    df_preview = df.head(10).round(2).astype(str).to_string(index=False)
    result = chain.invoke({
        "df_preview": df_preview,
        "history": "\n".join(chat_history),
        "input": user_input
    }, config={"configurable": {"session_id": "default"}})

    # 로그 저장
    import os
    from datetime import datetime

    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    # Append chat log with standardized format
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": user_input,
        "model_response": result
    }
    try:
        if os.path.exists("logs/chat_log.csv"):
            pd.DataFrame([log_entry]).to_csv("logs/chat_log.csv", mode='a', header=False, index=False)
        else:
            pd.DataFrame([log_entry]).to_csv("logs/chat_log.csv", mode='w', header=True, index=False)
    except Exception as e:
        print("Error saving log:", e)

    return result
