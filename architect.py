# 사용자가 선택한 기술 스택에 맞는 조언을 준비하는 파일

class DevelopmentArchitect:
    """
    다양한 개발 언어와 프레임워크에 대한 모범 사례 지식을 가진 설계자 클래스
    """
    def __init__(self,):
        # 각 기술별로 AI에게 줄 특별한 지침서를 미리 저장함
        self.stack_guides = {
            "Streamlit": "Python 기반의 데이터 대시보드 제작 전문가로서 답변하세요.",
            "React": "현대적인 함수형 컴포넌트와 Hooks를 사용하는 프론트엔드 전문가로서 답변하세요.",
            "Flutter": "Dart 언어와 위젯 구조에 능숙한 모바일 앱 개발 전문가로서 답변하세요.",
            "Flask": "가벼운 Python 웹 서버와 REST API 설계 전문가로서 답변하세요.",
            "HTML/CSS": "웹 표준과 반응형 디자인을 준수하는 퍼블리싱 전문가로서 답변하세요.",
            "Java": "객체 지향 원칙과 Spring Framework 지식을 갖춘 백엔드 전문가로서 답변하세요.",
            "JavaScript": "ES6+ 문법과 비동기 처리에 능숙한 풀스택 개발자로서 답변하세요."
        }

    def get_system_prompt(self, stack_name):
        """
        선택된 기술 이름에 해당하는 전문적인 지침 문구를 가져옴
        """
        # 만약 목록에 없는 기술이라면 일반적인 소프트웨어 엔지니어로 대답
        return self.stack_guides.get(stack_name, "경험 많은 소프트웨어 엔지니어로서 답변하세요.")