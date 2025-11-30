물론이죠. 요청하신 텍스트를 마크다운 형식으로 변환하여 복사하기 쉽게 해 드립니다.

````markdown
# 📁 Py-Local-Code-RAG: 기밀 코드를 위한 로컬 AI 코딩 도구

## 🌟 프로젝트 개요

**Py-Local-Code-RAG**는 회사의 민감한 코드를 외부 서버로 유출하지 않고, 사용자 **로컬 환경(에어갭 환경)**에서 LLM(Large Language Model)을 통해 코드베이스의 전체 맥락을 이해하고 코딩 작업을 지원하도록 설계된 개인용 코딩 어시스턴트입니다.

---

## 💡 핵심 목표

* **완벽한 보안**: 모든 데이터 처리, 임베딩, LLM 추론은 **사용자 로컬 머신에서만** 이루어집니다.
* **폴더 맥락 이해 (RAG)**: **검색 증강 생성(RAG)** 아키텍처를 사용하여, 대규모 코드베이스의 구조와 내용을 이해하고 질문에 답변합니다.
* **Python 기반**: VS Code 환경에 최적화된 모듈식 Python 도구입니다.

---

## 🧱 아키텍처 및 기술 스택

이 도구는 **하이브리드 RAG 아키텍처**를 사용하며, LLM 실행은 **Ollama**에 위임하여 개발 편의성을 높입니다.

| 영역 | 사용 기술 | 역할 |
| :--- | :--- | :--- |
| 핵심 언어 | **Python** | 전체 RAG 파이프라인 관리 |
| LLM 실행 환경 | **Ollama** | 로컬 LLM 서버 (예: Code Llama 8B) |
| RAG 프레임워크 | **LangChain** | 문서 로드, 분할, 검색, 체인 오케스트레이션 |
| 벡터 데이터베이스 | **ChromaDB** | 코드 청크의 벡터 임베딩 저장소 |
| 임베딩 모델 | **BAAI/bge-small-en-v1.5** | 코드의 의미를 벡터로 변환 |

---

## 🛠️ 설치 및 사용법 (단계별 가이드)

### 1. 전제 조건

* **Python 3.11 이상**
* **Ollama 설치 및 모델 다운로드**:
    * Ollama 공식 웹사이트에서 설치합니다.
    * 터미널에서 원하는 모델을 다운로드합니다 (예: `ollama pull codellama:8b`).

### 2. 프로젝트 초기 설정 및 의존성 설치

```bash
# 1. 가상 환경 생성 및 활성화 (권장)
python -m venv venv
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 2. 의존성 설치 (requirements.txt 사용)
pip install -r requirements.txt
````

### 3\. 1단계: 코드베이스 색인 (Index)

분석할 프로젝트 폴더 경로를 지정하고 `code_indexer.py`를 실행하여 벡터 데이터베이스를 생성합니다.

```bash
# 주의: /path/to/your/company/code 를 실제 프로젝트 경로로 변경해야 합니다.
python code_indexer.py /path/to/your/company/code
```

> 결과로 `chroma_db/` 폴더가 생성됩니다. (이 폴더는 `.gitignore`에 의해 GitHub에 올라가지 않습니다.)

### 4\. 2단계: 질의응답 (QA) 시작

Ollama 서버가 실행 중인 상태에서 (터미널에서 `ollama run codellama`를 실행하거나, Ollama 앱이 백그라운드 실행 중인지 확인) `code_qa_tool.py`를 실행하여 AI와 대화를 시작합니다.

```bash
python code_qa_tool.py
```