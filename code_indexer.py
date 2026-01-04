# 내 프로젝트 폴더를 읽어서 벡터 DB와 그래프 DB에 지식으로 저장하는 파일입니다.

import os
import ast
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from graph_manager import CodeGraphManager

class CodebaseIndexer:
    """
    프로젝트의 모든 파일을 읽어서 검색 가능한 형태로 가공하고 저장하는 클래스
    """
    def __init__(self, db_path, embed_model):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name= embed_model)

    def index_project(self, root_dir, user_id, project_name):
        """
        폴더 안의 모든 파일을 읽어 VectorDB와 GraphDB를 구축함
        """
        all_docs = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith((".py", ".js", ".java", ".html", ".css", ".md")):
                    path = os.path.join(root, file)
                    try:
                        loader = TextLoader(path, encoding= 'utf-8')
                        docs = loader.load()
                        for d in docs:
                            d.metadata['source'] = os.path.relpath(path, root_dir)
                        all_docs.extend(docs)
                    except: continue        # 읽지 못하는 파일 무시하고 넘어감.

        # 1. VectorDB (단어 유사도 기반)
        splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 200)
        chunks = splitter.split_documents(all_docs)

        # 사용자별 폴더 구분해서 VectorDB 생성
        persist_dir = os.path.join(self.db_path, user_id, project_name)
        vector_db = Chroma.from_documents(chunks, self.embeddings, persist_directory= persist_dir)
        vector_db.persist()

        # 2. GraphDB (Neo4j - 관계 기반)
        # Neo4j 서버가 켜져 있어야 하며, 아이디/비번이 맞아야 함.
        graph = CodeGraphManager("bolt://localhost:7687", "neo4j", "password")
        for doc in all_docs:
            source = doc.metadata['source']
            if source.endswith(".py"):
                try:
                    tree = ast.parse(doc.page_content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for n in node.names: graph.add_relation(source, n.name)
                except: pass
        graph.close()

        return len(chunks)