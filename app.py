import os
import csv
import time
import warnings
import socket
from datetime import datetime
import streamlit as st

# 불필요한 경고 메세지 숨기기
warnings.filterwarnings("ignore")

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
DEFAULT_MODEL = "qwen2.5-coder:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
FEEDBACK_FILE = "rag_feedback.csv"

# 페이지 설정
st.set_page_config(page_title="Agentic Code Assistant", layout="wide")

# 1. 로그인 세션
# ----------
if "user_id" not in st.session_state:
    st.session_state.user_id = None


def login_ui():
    st.title("AI Co-Developer 로그인")
    st.markdown("자신의 ID(또는 팀명)를 입력하여 전용 작업 공간에 접속하세요.")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input(
            "사용자 ID", placeholder="예: dev_kim", key="login_input"
        )
    with col2:
        st.write("")  # 줄맞춤
        st.write("")
        if st.button("접속하기", type="primary"):
            if user_input:
                st.session_state.user_id = user_input
                st.rerun()
            else:
                st.warning("ID를 입력해주세요.")


# 로그인이 안 되어 있으면 로그인 화면만 표시하고 중단
if not st.session_state.user_id:
    login_ui()
    st.stop()


# 2. 메인 어플리케이션 (로그인 후)
# ----------
CURRENT_USER = st.session_state.user_id

with st.sidebar:
    st.info(f"접속자: {CURRENT_USER}")
    if st.button("로그아웃"):
        st.session_state.user_id = None
        st.rerun()
    st.divider()

st.markdown(f"### Agentic Co-Developer: {CURRENT_USER}")


# 로컬 IP 확인 함수
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


# 유틸리티 함수
def get_existing_projects(user_id):
    """현재 접속한 사용자의 폴더(Chroma_db/user_id)만 조회합니다."""
    user_path = os.path.join(BASE_DB_PATH, user_id)
    if not os.path.exists(user_path):
        return []
    # 폴더이면서 숨김 파일이 아닌 것들만 리스트업
    return sorted(
        [
            d
            for d in os.listdir(user_path)
            if os.path.isdir(os.path.join(user_path, d)) and not d.startswith(".")
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
    st.header("환경 설정")

    # 1. 모델 선택
    model_options = ["qwen2.5-coder:7b", "llama3", "codellama", "mistral"]
    selected_model = st.selectbox(
        "LLM 모델 선택",
        model_options,
        index=0,
        help="Ollama에 설치된 모델 이름을 선택하세요. (기본값: qwen2.5-coder)",
    )

    st.divider()

    # 2. 프로젝트 선택
    st.subheader("프로젝트 관리")
    existing_projects = get_existing_projects(CURRENT_USER)

    tab1, tab2 = st.tabs(["내 프로젝트", "새로 추가"])

    project_name = None

    with tab1:
        if existing_projects:
            project_name = st.selectbox("프로젝트 선택", existing_projects)
            st.success(f"'{project_name}' 준비됨")
        else:
            st.info("학습된 프로젝트가 없습니다.")

    with tab2:
        new_project_name = st.text_input("새 프로젝트 이름", placeholder="my-project")
        new_root_path = st.text_input("실제 파일 경로", placeholder="C:/Work/MyProject")

        if st.button("학습 시작", type="primary"):
            if not new_project_name or not new_root_path:
                st.error("이름과 경로를 모두 입력하세요.")
            else:
                with st.spinner(f"'{new_project_name}' 학습 중"):
                    success, msg = embed_project(
                        new_root_path, new_project_name, CURRENT_USER
                    )
                    if success:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
        if new_project_name and not project_name:
            project_name = new_project_name

    st.divider()
    project_root_path = st.text_input(
        "파일 트리 경로(선택)", help="파일 구조 시각화를 위한 실제 경로"
    )

    if project_root_path and os.path.isdir(project_root_path):
        with st.expander("파일 구조"):
            st.code(generate_file_tree(project_root_path))

    # 대화 기록 초기화 버튼
    if st.button("대화 내용 지우기"):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"공유 주소: http://{get_local_ip()}:8501")


# RAG 에이전트 로드
@st.cache_resource
def load_agent(user_id, prj_name, model_name):
    """사용자별 프로젝트와 선택된 LLM 모델을 사용하여 에이전트를 로드합니다."""
    db_path = os.path.join(BASE_DB_PATH, user_id, prj_name)
    if not os.path.exists(db_path):
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    try:
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        # 에이전트 인스턴스 생성 및 그래프 빌드
        agent_instance = LocalRAGAgent(retriever, model_name, OLLAMA_BASE_URL)
        app_graph = agent_instance.build_graph()

        return app_graph
    except Exception as e:
        return str(e)


# 피드백 로깅 함수
def log_feedback(project, user, question, answer, rating, docs):
    file_exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Time",
                    "Project",
                    "User",
                    "Question",
                    "Answer",
                    "Rating",
                    "Context_Files",
                ]
            )

        # 문서 객체에서 소스만 추출
        sources = [d.metadata.get("source", "Unknown") for d in docs] if docs else []
        writer.writerow(
            [datetime.now(), project, user, question, answer, rating, str(sources)]
        )


# 메인 실행 로직
app_graph = None
current_tree = ""
is_ready = False

if project_name:
    # 선택된 모델을 인자로 전달
    result = load_agent(CURRENT_USER, project_name, selected_model)

    # LangGraph 컴파일된 객체인지 확인 (Callable 하거나 invoke 메서드가 있어야 함)
    if result and hasattr(result, "invoke"):
        app_graph = result
        is_ready = True
        if project_root_path:
            current_tree = generate_file_tree(project_root_path)
    else:
        system_msg = "프로젝트를 선택해주세요."

# 채팅 UI

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
        if not is_ready:
            st.error("프로젝트가 선택되지 않았거나 로드할 수 없습니다.")
        else:
            try:
                with st.spinner(f"{selected_model}가 생각하고 검증하는 중"):
                    # LangGraph 실행
                    inputs = {
                        "question": prompt,
                        "project_name": project_name,
                        "file_tree": current_tree,
                    }

                    final_state = app_graph.invoke(inputs)
                    answer = final_state.get(
                        "generation", "답변을 생성하지 못했습니다."
                    )

                    st.markdown(answer)

                    # 검증된 문서만 근거로 표시
                    valid_docs = final_state.get("documents", [])
                    if valid_docs:
                        with st.expander(f"검증된 근거 문서 ({len(valid_docs)}개)"):
                            for doc in valid_docs:
                                st.caption(f"{doc.metadata.get('source', 'Unknown')}")
                                st.code(doc.page_content)
                    else:
                        st.caption(
                            "검색된 문서 중 관련성 높은 코드가 없어 일반적인 지식으로 답변했습니다."
                        )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                # 피드백 상태 저장
                st.session_state.last_interaction = {
                    "p": project_name,
                    "q": prompt,
                    "a": answer,
                    "d": valid_docs,
                }
                st.rerun()

            except Exception as e:
                st.error(f"오류: {e}")

# 피드백 UI
if (
    is_ready
    and "last_interaction" in st.session_state
    and st.session_state.last_interaction
):
    st.divider()
    cols = st.columns([1, 1, 6])
    last = st.session_state.last_interaction

    if cols[0].button("👍"):
        log_feedback(last["p"], CURRENT_USER, last["q"], last["a"], "Good", last["d"])
        st.toast("피드백 저장됨!")
        del st.session_state.last_interaction
        st.rerun()

    if cols[1].button("👎"):
        log_feedback(last["p"], CURRENT_USER, last["q"], last["a"], "Bad", last["d"])
        st.toast("피드백 저장됨!")
        del st.session_state.last_interaction
        st.rerun()
