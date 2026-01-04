# AI와의 대화를 위해 다양한 언어를 번역해주는 번역사 역할을 하는 파일

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

class LanguageTranslator:
    """
    사용자가 입력한 언어를 AI가 잘 이해하는 영어로 바꾸거나,
    AI의 영어 답변을 사용자가 원하는 언어로 다시 번역해주는 클래스
    """
    def __init__(self, model_name, base_url):
        # 번역을 수행할 AI 모델의 이름과 접속 주소를 설정합니다.
        self.llm = ChatOllama(model=model_name, base_url=base_url, temperature=0)

    def translate(self, text, target_lang, source_lang= "Auto"):
        """
        입력받은 문장을 목표 언어로 번역하여 결과물로 돌려줍니다.
        """
        prompt = PromptTemplate(
            template="""You are a professional technical translator.
            Translate the following text from {source_lang} to {target_lang}.
            If it's a technical term or code, keep the meaning precise.
            Output ONLY the translated text without any explanation.
        
            Text: {text}

            Translation:""",
            input_variables=["text", "source_lang", "target_lang"],
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"text": text}, "source_lang": source_lang, "target_lang": target_lang).strip()