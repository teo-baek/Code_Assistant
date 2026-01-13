# 웹 화면 메인 파일

import streamlit as st
import os
from streamlit_agraph import agraph, Node, Edge, Config

from brain_manager import BrainManager
from translator import LanguageTranslator
from architect import DevelopmentArchitect
from code_indexer import CodebaseIndexer
from rag_agent import AgenticBrain

from graph_manager import CodeGraphManager
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

class CodeAssistantUI:
    """
    웹 화면의 모든 버튼과 기능을 배치하고 사용자와 소통하는 클래스
    사용자가 AI의 생각을 시각적으로 확인하고, 코드를 직접 수정/승인하는 인터페이스
    """
    def __init__(self):
        # 기본 서버 주소와 설정값을 정함
        self.OLLAMA_URL = "http://localhost:11434"
        self.DB_PATH = "./chroma_db"
        self.EMBED_MODEL = "BAAI/bge-small-en-v1.5"
        self.NEO4J_URI = "bolt://localhost:7687"        # Neo4j 주소

        self.brain_mgr = BrainManager(self.OLLAMA_URL)
        self.architect = DevelopmentArchitect()
        self.graph_mgr = CodeGraphManager(self.NEO4J_URI, "neo4j", "password")

    def show_graph_viz(self, file_name):
        """
        Neo4j 데이터를 읽어와서 특정 파일의 영향 범위를 그래프 그림으로 보여줌
        """
        st.subheader(f"'{file_name}' 관련 의존성 지도")

        # Neo4j에서 관계 데이터를 가져옴
        relations = self.graph_mgr.get_context_map(file_name)

        nodes = []
        edges = []

        # 중심 노드 추가
        nodes.append(Node(id=file_name, label= file_name, size= 25, color= "#005088"))

        for rel in relations:
            parts = rel.split(" --")
            neighbor = parts[1].split("-- ")[1]
            rel_type = parts[1].split("(")[1].split(")")[0]

            # 이웃 점과 연결 선을 추가합니다.
            nodes.append(Node(id= neighbor, label= neighbor, size= 15, color="#11CAA0"))
            edges.append(Edge(source= file_name, target= neighbor, label= rel_type))
        
        # 그래프 설정
        config = Config(width= 800, height= 400, directed= True, nodeHighlightBehavior= True, highlightColor= "#F3F0DF", collapsible= True)

        # 화면에 그래프를 그림
        agraph(nodes= nodes, edges= edges, config= config)

    def run(self):
        """웹 화면을 구성하고 프로그램을 실행함"""
        st.set_page_config(page_title= "GrowCode", layout= "wide")
        st.title("🚀 GrowCode")
        
        # 1. 로그인 
        if "user_id" not in st.session_state:
            st.session_state.user_id = st.text_input("사용자 ID를 입력하고 엔터를 누르세요.")
            if not st.session_state.user_id: st.stop()
            st.rerun()
        
        # 2. 사이드바 - 설정
        with st.sidebar:
            st.header("⚙️ 환경 설정")
            # 내 컴퓨터의 Ollama 모델 목록을 가져옴
            models = self.brain_mgr.get_available_models()
            selected_model = st.selectbox("사용할 모델 선택", models if models else ["모델을 찾을 수 없음"])

            # 답변 받을 언어 선택
            selected_lang = st.selectbox("답변 언어 선택", ["Korean", "English", "Japanese", "Chinese"])

            # 개발할 기술 스택 선택
            selected_stack = st.selectbox("개발 기술 스택 선택", ["Streamlit", "React", "Flutter", "Flast", "HTML/CSS", "Java", "JavaScript"])

            st.divider()
            if st.button("그래프 DB 초기화"):
                self.graph_mgr.reset_graph()
                st.toast("Neo4j 데이터가 초기화되었습니다.")
            
            # 메인 화면: 2분할 (왼쪽: 채팅/시각화, 오른쪽: 코드 리뷰)
            col_chat, col_review = st.columns([1, 1])

            with col_chat:
                st.subheader("GrowCode")
                if "messages" not in st.session_state: st.session_state.messages = []
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
                
                if prompt := st.chat_input("작업을 지시하세요"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.rerun()
            
            # 질문이 새로 들어왔을 때의 처리 로직
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                user_prompt = st.session_state.messages[-1]["content"]

                # 1. 번역 및 검색 준비
                translator = LanguageTranslator(selected_model, self.OLLAMA_URL)
                en_query = translator.translate(user_prompt, "English")

                db_dir = os.path.join(self.DB_PATH, st.session_state.user_id, "default_project")
                if os.path.exists(db_dir):

            st.subheader("프로젝트 지식 추가")
            p_name = st.text_input("프로젝트 별명")
            p_path = st.text_input("폴더 실제 경로")
            if st.button("지식 저장 시작"):
                indexer = CodebaseIndexer(self.DB_ROOT, self.EMBED_MODEL)
                count = indexer.index_project(p_path, st.session_state.user_id, p_name)
                st.success(f"{count} 개의 지식 조각 저장 완료")

        # 3. 채팅 화면
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        # 4. 사용자 질문 입력
        if prompt := st.chat_input("질문을 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            # AI가 답변을 준비하는 과정
            with st.chat_message("assistant"):
                # (1) 번역 준비: 사용자 질문을 영어로 번역
                translator = LanguageTranslator(selected_model, self.OLLAMA_URL)
                en_query = translator.translate(prompt, "English")
                st.caption(f"추론용 번역: {en_query}")

                # (2) DB 연결: 지정된 지식을 찾을 준비
                db_dir = os.path.join(self.DB_PATH, st.session_state.user_id, "default_project")        # 예시 default_project
                if os.path.exists(db_dir):
                    embed_ai = HuggingFaceBgeEmbeddings(model_name= self.EMBED_MODEL)
                    vector_db = Chroma(persist_directory= db_dir, embedding_function=embed_ai)
                    retriever = vector_db.as_retriever(search_kwargs= {"k": 5})

                    # (3) 에이전트 실행: 질문에 답하기 위한 지식 검색 및 추론
                    brain = AgenticBrain(selected_model, self.OLLAMA_URL, retriever)
                    flow = brain.build_workflow()

                    # 설계자로부터 해당 기술에 맞는 전문 지침을 가져옴
                    sys_prompt = self.architect.get_system_prompt(selected_stack)

                    with st.spinner("생각하는 중"):
                        # 사고 흐름 실행
                        final_state = flow.invoke({
                            "question": en_query,
                            "system_prompt": sys_prompt,
                            "stack": selected_stack
                        })
                        en_answer = final_state["answer"]

                        # (4) 재번역: AI의 답변을 사용자 언어로 번역
                        final_answer = translator.translate(en_answer, selected_lang)
                        st.markdown(final_answer)

                        # 대화 기록에 저장
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})
                else:
                    st.error("학습된 프로젝트가 없습니다. 사이드바에서 먼저 학습시켜주세요.")

if __name__ == "__main__":
    app = CodeAssistantUI()
    app.run()