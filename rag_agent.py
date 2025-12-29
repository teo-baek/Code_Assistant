import os
import csv
from typing import Dict, TypedDict, List, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langgraph.graph import END, StateGraph
from graph_builder import CodeGraphBuiler

FEEDBACK_FILE = "rag_feedback.csv"


# 상태(State) 정의
class GraphState(TypedDict):
    """
    그래프의 각 노드 간에 전달되는 데이터 상태입니다.
    """

    question: str
    generation: str
    documents: List[Any]
    project_name: str
    file_tree: str
    db_path: str


# 에이전트 클래스
class LocalRAGAgent:
    def __init__(self, retriever, llm_model_name, base_url):
        self.retriever = retriever
        # JSON 출력을 강제하기 위해 format= "json" 옵션 사용 (Ollama 기능)
        self.llm_json = ChatOllama(
            model=llm_model_name, base_url=base_url, format="json", temperature=0
        )
        self.llm = ChatOllama(model=llm_model_name, base_url=base_url, temperature=0)
        self.graph_builder = CodeGraphBuiler()

    def get_successful_examples(self, k=3):
        """
        퓨샷 러닝: 성공적인 예시 가져오기
        """
        if not os.path.exists(FEEDBACK_FILE):
            return ""
        examples = []
        try:
            with open(FEEDBACK_FILE, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)[::-1]
                for row in rows:
                    if row.get("Rating") == "Good":
                        q = row.get("Question", "").strip()
                        a = row.get("Answer", "").strip()
                        if q and a:
                            examples.append(f"Q: {q}\nA: {a}")
                    if len(examples) >= k:
                        break
        except:
            return ""
        if not examples:
            return ""
        formatted = "\n\n".join(examples)
        return f"\n[참고할 모범 답변 패턴]:\n{formatted}"

    # 1. 문서 검색 노드
    def retrieve(self, state: GraphState):
        print("RETRIEVE")
        question = state["question"]

        # 1차: 벡터 DB 검색 (기존 방식)
        documents = self.retriever.invoke(question)
        print(f"    - 벡터 검색: {len(documents)}개 문서 발견")

        # 2차: 그래프 확장
        # 상태에 저장된 db_path를 통해 그래프 파일을 로드합니다.
        db_path = state.get("db_path", "")
        graph_file = os.path.join(db_path, "project_graph.pkl")

        if self.graph_builder.load(graph_file):
            print(" - 그래프 데이터 로드 성공. 연관 파일 탐색 중.")
            related_docs = []

            # 벡터 검색으로 찾은 문서들의 연관 파일을 그래프에서 찾습니다.
            for doc in documents:
                source_file = doc.metadata.get("source")
                if source_file:
                    # 그래프에서 이 파일과 연결된 파일들을 가져옵니다.
                    related_files = self.graph_builder.get_related_files(source_file)
                    for rel_file in related_files:
                        # 실제로는 해당 파일의 내용을 다시 로드해야 하지만,
                        # 여기서는 컨텍스트에 메타데이터를 추가하는 방식으로 경량화합니다.
                        # 또는 간단한 파일명만 리스트에 추가하여 LLM에게 힌트를 줍니다.
                        # 여기서는 가상의 문서 객체를 만들어 추가합니다.
                        related_docs.append(
                            f"[Graph Hint] '{source_file}' 파일은 '{rel_file}' 파일과 연결되어 있습니다. "
                        )

            if related_docs:
                graph_context = "\n".join(related_docs)
                # 첫 번째 문서에 몰아서 붙여줍니다.
                documents[
                    0
                ].page_content += f"\n\n[구조적 연관 정보]: \n{graph_context}"
                print(f"    - 그래프 힌트 {len(related_docs)}개 추가됨")

        return {"documents": documents, "question": question}

    # 2. 문서 평가 노드
    def grade_documents(self, state: GraphState):
        print("CHECK RELEVANCE")
        question = state["question"]
        documents = state["documents"]

        # 평가 프롬프트
        prompt = PromptTemplate(
            template="""당신은 검색된 코드 문서가 사용자 질문과 관련이 있는지 평가하는 채점관입니다.

            문서 내용:
            {document}

            사용자 질문:
            {question}

            문서가 질문의 키워드를 포함하거나 의미적으로 관련이 있다면 'yes', 아니면 'no'로 평가하세요.
            반드시 JSON 형식으로 출력하세요: {{"score": "yes"}} 또는 {{"score": "no"}}
            """,
            input_variables=["question", "document"],
        )

        grader = prompt | self.llm_json | JsonOutputParser()

        filtered_docs = []
        for d in documents:
            try:
                score = grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                if score["score"] == "yes":
                    print("     - GRADE: RELEVANT (관련 있음)")
                    filtered_docs.append(d)
                else:
                    print("     - GRADE: NOT RELEVANT (관련 없음 - 제거됨)")

            except Exception:
                continue

        return {"documents": filtered_docs, "question": question}

    # 3. 답변 생성 노드
    def generate(self, state: GraphState):
        print("GENERATE")
        question = state["question"]
        document = state["documents"]
        file_tree = state.get("file_tree", "")

        # 문서가 하나도 없으면 검색 실패 메세지 반환
        if not document:
            return {
                "generation": "죄송합니다. 제공해주신 프로젝트 코드 내에서 관련 정보를 찾을 수 없습니다. 질문을 조금 더 구체적으로 해주시겠어요?"
            }

        # 문맥 합치기
        context = "\n\n".join([d.page_content for d in document])

        prompt = PromptTemplate(
            template="""당신은 Tech Lead로서 답변합니다.

            [프로젝트 구조]:
            {file_tree}

            [검색된 코드 맥락]:
            {context}

            [질문]:
            {question}

            위 맥락을 바탕으로 한국어로 상세히 답변하세요. 맥락에 없는 내용은 지어내지 마세요.
            """,
            input_variables=["question", "context", "file_tree"],
        )

        # 일반 LLM 사용
        chain = prompt | self.llm | StrOutputParser()
        generation = chain.invoke(
            {"question": question, "context": context, "file_tree": file_tree}
        )

        return {"generation": generation}

    # 그래프 구축
    def build_graph(self):
        workflow = StateGraph(GraphState)

        # 노드 정의
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)

        # 엣지 연결
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()
