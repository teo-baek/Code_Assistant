# AI가 질문을 받고 답변을 생성하는 전체 워크플로우를 관리하는 파일

from typing import TypedDict, List, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langgraph.graph import END, StateGraph

class AgentState(TypedDict):
    """
    AI가 사고하는 과정에서의 상태 정의
    """
    question: str
    context: str
    answer: str
    stack: str
    system_prompt: str

class AgenticBrain:
    """
    질문을 분석하고 지식을 찾아 답변을 만드는 클래스
    """
    def __init__(self, model_name, base_url, retriever):
        # 대화에 사용한 AI와 지식을 찾아올 검색기 준비
        self.llm = ChatOllama(model= model_name, base_url= base_url, temperature= 0)
        self.retriever = retriever

    def search_node(self, state: AgentState):
        """[1단계] 질문관 관련된 코드를 검색기에서 찾아옴"""
        print("지식 검색 중")
        docs = self.retriever.invoke(state["question"])
        context = "\n\n".join([d.page_content for d in docs])
        return {"context": context}
    
    def answer_node(self, state: AgentState):
        """[2단계] 찾은 지식과 역할을 바탕으로 답변을 작성함"""
        print("답변 작성 중")
        prompt = PromptTemplate(
            template="""{system_prompt}

            Below is the context code from the project:
            {context}

            Question: {question}

            Answer the question based strictly on the context provided.
            """,
            input_variables= ["system_prompt", "context", "question"]
        )
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke(state)
        return {"answer": answer}
    
    def build_workflow(self):
        """AI의 사고 흐름을 하나로 연결함"""
        flow = StateGraph(AgentState)
        
        flow.add_node("search", self.search_node)
        flow.add_node("answer", self.answer_node)
        
        flow.set_entry_point("search")
        
        flow.add_edge("search", "answer")
        flow.add_edge("answer", END)

        return flow.compile()