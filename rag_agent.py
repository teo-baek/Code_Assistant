from typing import Dict, TypedDict, List, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langgraph.graph import END, StateGraph


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


# 에이전트 클래스
class LocalRAGAgent:
    def __init__(self, retriever, llm_model_name, base_url):
        self.retriever = retriever
        # JSON 출력을 강제하기 위해 format= "json" 옵션 사용 (Ollama 기능)
        self.llm_json = ChatOllama(
            model=llm_model_name, base_url=base_url, format="json", temperature=0
        )
        self.llm = ChatOllama(model=llm_model_name, base_url=base_url, temperature=0)

    # 1. 문서 검색 노드
    def retrieve(self, state: GraphState):
        print("RETRIEVE")
        question = state["question"]
        documents = self.retriever.invoke(question)
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
