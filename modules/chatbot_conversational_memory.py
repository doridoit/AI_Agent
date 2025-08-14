from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import re

def create_memory_conversational_chain(llm, df_head: str) -> RunnableWithMessageHistory:
    """
    Streamlit 기반 메시지 히스토리를 사용하는 대화형 체인 생성.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 Python 기반 데이터 분석 AI야. 주어진 표를 기반으로 사용자의 질문에 정확하게 응답해야 해."),
        ("human", "다음은 데이터프레임 일부야:\n{df_head}"),
        ("human", "질문: {question}")
    ])

    base_chain: RunnableSequence = prompt | llm | StrOutputParser()
    
    chat_history = StreamlitChatMessageHistory()

    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    return chain_with_history

def extract_analysis_intent_and_column(user_input, df_columns):
    # Normalize user input to lower case
    text = user_input.lower()

    # Define common synonyms for operations
    mean_keywords = ['mean', 'average', '평균']
    max_keywords = ['maximum', 'max', '최대']
    min_keywords = ['minimum', 'min', '최소']
    sum_keywords = ['sum', 'total', '합계']

    operation = None
    if any(k in text for k in mean_keywords):
        operation = 'mean'
    elif any(k in text for k in max_keywords):
        operation = 'max'
    elif any(k in text for k in min_keywords):
        operation = 'min'
    elif any(k in text for k in sum_keywords):
        operation = 'sum'

    # Try to match column names
    selected_column = None
    for col in df_columns:
        if col.lower() in text:
            selected_column = col
            break

    return operation, selected_column
