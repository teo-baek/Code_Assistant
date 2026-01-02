from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

class TranslatorAgent:
    def __init__(self, llm_model_name, base_url):
        # 번역을 위한 LLM 인스턴스 생성
        self.llm = ChatOllama(model=llm_model_name, base_url=base_url, temperature=0)

    def translate_to_english(self, text):
        """
        한국어 텍스트를 영어로 번역합니다.
        코드 생성에 최적화된 기술적인 영어 표현을 사용하도록 유도합니다.
        """
        prompt = PromptTemplate(
            template="""You are a professional technical translator.
            Translate the following Korean text into English.
            The translation should be optimized for a code generation AI to understand.
            Use precise technical terminology.
            Do not include any explanations, only the translated text.

            Korean text: {text}

            English translation:""",
            input_variables=["text"],
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"text": text}).strip()

    def translate_to_korean(self, text, source_language="English"):
        """
        영어 텍스트를 한국어로 번역합니다.
        기술 용어는 원어 그대로 유지하거나 자연스러운 한국어 용어로 변환합니다.
        """
        prompt = PromptTemplate(
            template="""You are a professional technical translator.
            Translate the following text (which is in {source_language}) into Korean.
            Keep technical terms (like variable names, function names, library names) in their original English form if appropriate for a developer audience.
            Ensure the tone is professional and helpful.
            Do not include any explanations, only the translated text.

            Text to translate:
            {text}

            Korean translation:""",
            input_variables=["text", "source_language"],
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"text": text, "source_language": source_language}).strip()