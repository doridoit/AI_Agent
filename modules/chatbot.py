import os
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

llm = None
_API_INIT_ERROR = None
_CURRENT_MODEL_ID = None  # e.g., "google:gemini-1.5-pro"

load_dotenv()

def get_available_llms():
    providers = []
    if os.getenv("GOOGLE_API_KEY"):
        providers.append(("Gemini (Google)", "google:gemini-1.5-pro"))
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("OpenAI (GPT-4o-mini)", "openai:gpt-4o-mini"))
    return providers

def init_llm(provider_id: str):
    global llm, _API_INIT_ERROR, _CURRENT_MODEL_ID
    try:
        provider_id = (provider_id or '').strip()
        if not provider_id:
            raise ValueError("모델 식별자가 비어 있습니다.")

        if provider_id.startswith("google:"):
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY가 설정되지 않았습니다. .env 또는 환경변수에 추가하세요.")
            llm = ChatGoogleGenerativeAI(
              model=provider_id.split(":", 1)[1],
              temperature=0.2,
              google_api_key=api_key,
            )
        elif provider_id.startswith("openai:"):
            from langchain_openai import ChatOpenAI  # optional provider
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 환경변수에 추가하세요.")
            llm = ChatOpenAI(model=provider_id.split(":", 1)[1], temperature=0.2, api_key=api_key)
        else:
            raise ValueError(f"지원하지 않는 provider_id입니다: {provider_id}")

        _API_INIT_ERROR = None
        _CURRENT_MODEL_ID = provider_id
        return "ok"
    except Exception as e:
        _API_INIT_ERROR = str(e)
        _CURRENT_MODEL_ID = None
        llm = None
        return _API_INIT_ERROR

def get_status():
    """Return current LLM readiness and model info for UI."""
    return {
        "ready": llm is not None,
        "model": _CURRENT_MODEL_ID,
        "error": _API_INIT_ERROR,
    }

def ask_with_llm(prompt: str, df: pd.DataFrame):
    if llm is None:
        return f"LLM이 초기화되지 않았습니다: {_API_INIT_ERROR or '모델을 먼저 활성화하세요.'}"
    try:
        python_repl = PythonREPLTool(locals={"df": df})
        tools = [python_repl]
        # REACT prompt with required variables for create_react_agent
        REACT_PROMPT = """
        당신은 데이터 분석 전문가입니다.
        주어진 DataFrame 'df'를 참고하여 사용자의 질문에 답하세요.
        가능하면 Python 코드를 사용해 계산하고, 결과를 근거와 함께 설명하세요.

        [컨텍스트]
        - 첫 5행: {df_head}
        - 컬럼: {df_columns}
        - 데이터 타입: {df_dtypes}

        사용 가능한 도구:
        {tools}

        도구 이름 목록:
        {tool_names}

        작업 메모:
        {agent_scratchpad}

        질문: {input}
                """

        # create_react_agent 가 자동으로 공급하는 변수들(tools, tool_names, agent_scratchpad, input)을 남겨두고
        # df 관련 정보만 partial 로 고정 주입한다.
        base_template = PromptTemplate(
            input_variables=[
                "input",
                "tools",
                "tool_names",
                "agent_scratchpad",
                "df_head",
                "df_columns",
                "df_dtypes",
            ],
            template=REACT_PROMPT,
        ).partial(
            df_head=df.head().to_string(),
            df_columns=str(list(df.columns)),
            df_dtypes=df.dtypes.to_string(),
        )

        agent = create_react_agent(llm, tools, base_template)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        response = agent_executor.invoke({"input": prompt})
        if isinstance(response, dict):
            return response.get("output") or response.get("final_output") or str(response)
        return str(response)
    except Exception as e:
        return f"에이전트 실행 중 오류: {e}"