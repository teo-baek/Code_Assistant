import os
import csv
import time
from datetime import datetime
import streamlit as st

try:
    from code_indexer import embed_project
    from rag_agent import LocalRAGAgent
except ImportError:
    st.error("필수 모듈을 찾을 수 없습니다. 같은 폴더에 있는지 확인하세요.")
    st.stop()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 설정
BASE_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL_NAME = "qwen2.5-coder:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
FEEDBACK_FILE = "rag_feedback.csv"

# 페이지 설정
st.set_page_config(page_title="Agentic Code Assistant", layout="wide")
st.markdown(
    """
### Agentic Co-Developer: 
이 도구는 Agentic RAG 기술을 사용하여, 검색된 코드가 질문과 관련이 있는지 스스로 검증하고 답변합니다.
"""
)


# 유틸리티 함수
def get_existing_projects():
    """chroma_db 폴더를 스캔하여 학습된 프로젝트 목록을 반환합니다."""
    if not os.path.exists(BASE_DB_PATH):
        return []
    # 폴더이면서 숨김 파일이 아닌 것들만 리스트업
    return sorted(
        [
            d
            for d in os.listdir(BASE_DB_PATH)
            if os.path.isdir(os.path.join(BASE_DB_PATH, d)) and not d.startswith(".")
        ]
    )


# 함수: 파일 트리 생성 (Context Map)
def generate_file_tree(startpath):
    """프로젝트의 전체 지도를 그려주어, 개발자가 어디를 수정해야 할지 위치를 파악하게 돕습니다."""
    if not startpath or not os.path.exists(startpath):
        return "(경로가 설정되지 않았거나 유효하지 않습니다.)"
    startpath = os.path.abspath(startpath)
    tree_lines = []

    for root, dirs, files in os.walk(startpath):
        # 정렬하여 출력 순서 고정
        dirs.sort()
        files.sort()
        dirs[:] = [d for d in dirs if not d.startswith(".")]  # 숨김 폴더 제외

        # 상대 경로 계산으로 정확한 깊이 파악
        rel_path = os.path.relpath(root, startpath)
        if rel_path == ".":
            level = 0
            # 루트 폴더명 출력
            tree_lines.append(f"{os.path.basename(startpath)}/")
        else:
            level = rel_path.count(os.sep) + 1
            indent = "    " * (level - 1)
            folder_name = os.path.basename(root)
            # 하위 폴더명 출력 (들여쓰기 적용)
            tree_lines.append(f"{indent} {folder_name}/")

        # 파일 출력
        sub_indent = "    " * level
        for f in files:
            if not f.startswith("."):
                tree_lines.append(f"{sub_indent} {f}")

    return "\n".join(tree_lines) if tree_lines else "(빈 폴더입니다.)"


# 사이드바
with st.sidebar:
    st.header("프로젝트 선택")
    existing_projects = get_existing_projects()

    project_name = (
        st.selectbox("학습된 프로젝트", existing_projects)
        if existing_projects
        else None
    )

    st.divider()
    project_root_path = st.text_input(
        "파일 트리 경로(선택)", help="파일 구조 시각화를 위한 실제 경로"
    )

    if project_root_path and os.path.isdir(project_root_path):
        with st.expander("파일 구조"):
            st.code(generate_file_tree(project_root_path))

    # 새 프로젝트 학습 세션
    with st.expander("새 프로젝트 학습"):
        new_name = st.text_input("새 DB 이름")
        new_path = st.text_input("새 프로젝트 경로")
        if st.button("학습 시작"):
            if new_name and new_path:
                with st.spinner("학습 중"):
                    success, msg = embed_project(new_path, new_name)
                    if success:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)


# RAG 에이전트 로드
@st.cache_resource
def load_agent(prj_name):
    db_path = os.path.join(BASE_DB_PATH, prj_name)
    if not os.path.exists(db_path):
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 에이전트 인스턴스 생성 및 그래프 빌드
    agent_instance = LocalRAGAgent(retriever, OLLAMA_MODEL_NAME, OLLAMA_BASE_URL)
    app_graph = agent_instance.build_graph()

    return app_graph


# 메인 로직
app_graph = load_agent(project_name) if project_name else None
current_tree = generate_file_tree(project_root_path) if project_root_path else ""

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 표시
for msg in st.session_state.messages:
    with st.chat_message("role"):
        st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not app_graph:
            st.error("프로젝트를 선택해주세요.")
        else:
            try:
                with st.spinner("생각하고 검증하는 중 (Agentic workflow)"):
                    # LangGraph 실행
                    inputs = {
                        "question": prompt,
                        "project_name": project_name,
                        "file_tree": current_tree,
                    }

                    # 스트리밍 대신 invoke로 전체 실행 결과 받기
                    final_state = app_graph.invoke(inputs)
                    answer = final_state["generation"]

                    st.markdown(answer)

                    # 검증된 문서만 근거로 표시
                    valid_docs = final_state.get("documents", [])
                    if valid_docs:
                        with st.expander(f"검증된 근거 문서 ({{len(valid_docs)}})"):
                            for doc in valid_docs:
                                st.caption(f"{doc.metadata.get('source')}")
                                st.code(doc.page_content)
                    else:
                        st.caption(
                            "검색된 문서 중 관련성 높은 코드가 없어 일반적인 지식으로 답변했습니다."
                        )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"오류: {e}")
