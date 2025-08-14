### ** 프로젝트 소개**

**프로젝트명:** AI 기반 데이터 분석 에이전트 (Data Analysis AI Agent)

**요약:**
이 프로젝트는 Streamlit 웹 애플리케이션을 통해 사용자가 업로드한 데이터를 **탐색적 데이터 분석(EDA)**, **이상 탐지**, 그리고 **LLM 기반 챗봇**을 활용하여 대화형으로 분석할 수 있는 AI 에이전트입니다.

-----

### ** 주요 기능**

  * **데이터 업로드**: CSV 및 Excel 파일을 업로드할 수 있으며, Excel 파일의 경우 시트 및 데이터 시작 행을 선택할 수 있습니다.
  * **탐색적 데이터 분석 (EDA)**:
      * 기본 통계, 결측치, 컬럼 타입 분석
      * 단변량 분석: 선택한 수치형 컬럼에 대한 기초 통계 및 시각화
      * 다변량 분석: 수치형 컬럼 간의 상관관계 매트릭스 및 히트맵 시각화
  * **이상 탐지**: 로지스틱 회귀, XGBoost, 오토인코더 등 다양한 모델을 선택하여 데이터의 이상치를 탐지합니다.
  * **대화형 LLM 챗봇**:
      * LangChain을 기반으로 한 챗봇을 통해 자연어로 데이터에 대해 질문하고 답변을 받을 수 있습니다.
      * Google Gemini와 OpenAI 모델을 지원합니다.
      * 챗봇의 대화 기록은 CSV 파일로 저장됩니다.

-----

### **🛠️ 기술 스택**

  * **프레임워크**: Streamlit, LangChain
  * **언어**: Python
  * **데이터 처리**: Pandas, Numpy
  * **분석 모델**: scikit-learn, XGBoost, Keras
  * **시각화**: Matplotlib, Seaborn
  * **환경 설정**: python-dotenv

-----

### **시작하기**

#### **1. 환경 설정**

1.  프로젝트 레포지토리를 클론합니다.
    ```bash
    git clone [repository-url]
    cd [repository-name]
    ```
2.  가상 환경을 생성하고 활성화합니다.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  필요한 라이브러리를 설치합니다.
    ```bash
    pip install streamlit pandas openpyxl scikit-learn xgboost tensorflow langchain langchain-google-genai langchain-openai python-dotenv seaborn matplotlib chardet numpy
    ```

#### **2. API 키 설정**

LLM을 사용하기 위해 `.env` 파일을 생성하고 아래와 같이 API 키를 입력합니다.
`GOOGLE_API_KEY` 또는 `OPENAI_API_KEY` 중 하나를 선택하여 사용하면 됩니다.

```
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

#### **3. 애플리케이션 실행**

터미널에서 아래 명령어를 실행하여 웹 애플리케이션을 시작합니다.

```bash
streamlit run main.py
```

-----

### **프로젝트 구조**

```
ai.agent_proto/
├── main.py                    # 애플리케이션의 메인 실행 파일
├── .env                       # API 키 및 환경 변수
├── logs/
│   └── chat_log.csv           # 챗봇 대화 로그
├── modules/
│   ├── components/            # 재사용 가능한 공통 컴포넌트
│   │   └── llm_selector.py      # LLM 모델 선택 및 초기화 로직
│   ├── processing/            # 데이터 분석 및 처리 기능
│   │   ├── eda.py               # EDA 로직
│   │   └── anomaly_model.py     # 이상 탐지 로직
│   └── chatbot/               # 챗봇 관련 기능
│       ├── prompts.py           # LLM 프롬프트 템플릿
│       └── conversational_chatbot.py # 대화형 챗봇 로직
└── ...
```

-----

### **💡 향후 개선 사항**

  * **RAG(Retrieval Augmented Generation) 도입**: 대용량 데이터셋 분석 시, RAG 기술을 적용하여 LLM의 답변 정확도를 높입니다.
  * **데이터 시각화 강화**: EDA 섹션에서 더 다양한 차트와 그래프 옵션을 제공합니다.
  * **로그 관리 기능 개선**: 챗 로그 관리 및 시각화 기능을 추가합니다.
  * **프롬프트 엔지니어링**: LLM의 답변 품질을 높이기 위한 프롬프트 최적화 작업을 진행합니다.
