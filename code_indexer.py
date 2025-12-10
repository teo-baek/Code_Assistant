import os
import sys
import argparse
import shutil
import warnings

# 경고 메세지 숨기기
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 1. 전역 설정
# 벡터 DB가 저장될 로컬 디렉토리
CHROMA_DB_PATH = "./chroma_db"
# 사용할 로컬 임베딩 모델의 이름.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# 분석할 파일 확장자 정의
# 분석 대상이 아닌 파일(예: 이미지, 바이너리)은 제외
CODE_EXTENSIONS = (
    # Python / Web
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".html",
    ".css",
    ".vue",
    ".json",
    # Java / Kotlin / Android
    ".java",
    ".kt",
    ".gradle",
    ".properties",
    # C / C++ / C#
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    # iOS / Swift
    ".swift",
    ".m",
    ".mm",
    # Go / Rust / PHP / Ruby
    ".go",
    ".rs",
    ".php",
    ".rb",
    # Shell / Config / Docs
    ".sh",
    ".bat",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".md",
    ".txt",
    ".sql",
    ".dockerfile",
)


def load_documents(root_dir: str):
    """
    프로젝트 폴더를 탐색하여 문서 로드
    """
    print(f"[{root_dir} 폴더 분석을 시작합니다.]")

    documents = []
    # os.walk를 사용하여 폴더를 재귀적으로 탐색.
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 숨김 폴더(.git, .venv, .vscode 등)는 탐색에서 제외.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for file in filenames:
            # 숨김 파일(.gitignore, .env 등)은 로드에서 제외
            if file.startswith("."):
                continue
            if file.endswith(CODE_EXTENSIONS):
                filepath = os.path.join(dirpath, file)
                try:
                    # 1차 시도: UTF-8 (자동감지)
                    loader = TextLoader(
                        filepath, encoding="utf-8", autodetect_encoding=True
                    )
                    docs = loader.load()
                except Exception:
                    try:
                        # 2차 시도: CP949 (한글 윈도우 호환)
                        loader = TextLoader(filepath, encoding="cp949")
                        docs = loader.load()
                    except Exception:
                        try:
                            # 3차 시도: Latin-1 (바이너리성 텍스트 강제 로드)
                            loader = TextLoader(filepath, encoding="latin-1")
                            docs = loader.load()
                        except Exception as e:
                            continue

                for doc in docs:
                    # 상대 경로를 메타데이터로 저장
                    doc.metadata["source"] = filepath.replace(root_dir, "").lstrip(
                        os.sep
                    )
                    documents.append(doc)
    print(f"총 {len(documents)}개의 코드 파일을 로드했습니다.")
    return documents


def index_codebase(documents: list[Document], project_name: str):
    """
    로드된 코드를 벡터화하여 저장합니다.
    """

    # 1. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    # 2. 임베딩 모델
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 3. 저장 경로 (프로젝트별 격리)
    persist_dir = os.path.join(CHROMA_DB_PATH, project_name)

    # Clean Build: 기존 데이터가 있으면 삭제하고 새로 생성
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            print(f"경고: 기존 DB 삭제 실패 ({e})")

    # 4. 벡터 DB 생성 및 저장
    print("벡터 데이터베이스 저장 중")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)
    # Chroma 최신 버전은 db.persist()가 자동 수행되지만, 명시적으로 남겨둠
    try:
        db.persist()
    except:
        pass

    return len(texts)


# 외부 호출용 래퍼 함수
def embed_project(root_dir, project_name):
    """
    Streamlit 등 외부 앱에서 호출하기 위한 통합 함수.
    성공 여부와 메시지를 반환합니다.
    """
    try:
        if not os.path.isdir(root_dir):
            return False, f"❌ 경로가 유효하지 않습니다: {root_dir}"

        docs = load_documents(root_dir)
        if not docs:
            return (
                False,
                "⚠️ 로드된 파일이 없습니다. 경로 내에 소스 코드가 있는지 확인하세요.",
            )

        chunk_count = index_codebase(docs, project_name)
        return (
            True,
            f"✅ 학습 완료! 총 {len(docs)}개 파일, {chunk_count}개 청크",
        )
    except Exception as e:
        return False, f"❌ 오류 발생: {str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="프로젝트 코드를 AI에게 학습시키는 도구"
    )
    parser.add_argument(
        "project_path", type=str, help="분석할 프로젝트 폴더의 절대 또는 상대 경로"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default",
        help="프로젝트 식별 이름 (기본값: default)",
    )
    args = parser.parse_args()

    # 입력된 경로를 절대 경로로 변환하여 파일 로딩의 안정성을 높입니다.
    root = os.path.abspath(args.project_path)
    name = args.name if args.name != "default" else os.path.basename(root)

    print(f"'{name}' 고도화 학습 시작.")
    success, msg = embed_project(root, name)
    print(msg)
