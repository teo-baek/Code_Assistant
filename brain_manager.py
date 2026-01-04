# 로컬에 설치된 Ollama AI 모델들을 관리하고 선택할 수 있게 돕는 파일입니다.

import requests

class BrainManager:
    """
    내 로컬 컴퓨터의 Ollama 서버에 어떤 모델들이 있는지 확인하고 관리하는 클래스
    """
    def __init__(self, base_url):
        self.base_url = base_url

    def get_available_models(self):
        """
        Ollama 서버에 접속해서 현재 다운로드되어 있는 모델 목록을 리스트로 가져옴.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return [m['name'] for m in response.json().get('models', [])]
            return []
        except:
            return []