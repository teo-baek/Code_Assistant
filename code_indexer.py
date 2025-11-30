import os
import sys
import argparse
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.java', '.c', '.cpp', 
    '.h', '.hpp', '.go', '.rs', '.php', '.md', '.json', '.yaml', '.yml', '.txt'
)

def load_documents(root_dir: str) -> list[Document]:
    """
    주어진 폴터에서 지정된 확장자의 모든 코드를 재귀적으로 로드.
    """
    print(f"[{root_dir} 폴더에서 코드 파일을 로드합니다.]")

    documents = []
    # os.walk를 사용하여 폴더를 재귀적으로 탐색.
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 숨김 폴더(.git, .venv, .vscode 등)는 탐색에서 제외.
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]

        for file in filenames:
            # 숨김 파일(.gitignore, .env 등)은 로드에서 제외
            if file.startswith('.'):
                continue

            if file.endswith(CODE_EXTENSIONS):
                filepath = os.path.join(dirpath, file)
                try:
                    # LangChain의 TextLoader를 사용하여 파일을 로드.
                    loader = TextLoader(filepath, encoding= 'utf-8')
                    docs = loader.load()
                    for doc in docs:
                        # 메타데이터에 파일 경로를 저장하여 검색 후 출처를 명시할 수 있게 함.
                        doc.metadata['source'] = filepath.replace(root_dir, '').lstrip(os.sep)
                        documents.append(doc)
                except Exception as e:
                    # 파일 인코딩 오류 등을 대비한 예외 처리
                    print(f"경고: {filepath} 파일을 로드할 수 있습니다. 오류: {e}")

    print(f"총 {len(documents)}개의 파일이 로드되었습니다.")
    return documents