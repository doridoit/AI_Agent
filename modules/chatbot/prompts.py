from langchain.prompts import ChatPromptTemplate

CHATBOT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "너는 Python 기반 데이터 분석 AI야. 사용자의 질문에 정확하게 답변해야 해."),
    ("human", "다음은 데이터의 일부 미리보기야:\n{df_preview}"),
    ("human", "이전 대화 기록은 다음과 같아:\n{history}"),
    ("human", "질문: {input}")
])