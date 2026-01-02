import os
import ast
import networkx as nx
import pickle


class CodeGraphBuiler:
    """
    프로젝트 내의 코드 파일들이 서로 어떻게 연결되어 있는지를 분석하여 그래프로 만드는 클래스입니다.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def _parse_imports(self, file_path):
        """
        하나의 파이썬 파일을 열어서, 이 파일이 어떤 다른 파일들을 import(참조)하는지 찾아냅니다.
        """
        imports = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # ast.parse는 소스 코드를 읽어서 컴퓨터가 이해하기 쉬운 트리 구조로 바꿉니다.
                tree = ast.parse(f.read())

            # 트리 구조를 순회하면서 'Import'나 'ImportFrom' 구문을 찾습니다.
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except Exception:
            pass
        return imports

    def build_graph(self, root_dir):
        """
        프로젝트 폴더 전체를 돌면서 모든 파일의 관계도를 그립니다.
        """
        print(f"그래프 구조 분석 시작: {root_dir}")

        # 1. 모든 파일을 노드(점)로 등록합니다.
        file_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, root_dir).replace("\\", "/")
                    file_paths.append((full_path, rel_path))
                    self.graph.add_node(rel_path, type="file")

        # 2. 파일 내용을 읽어서 연결 선(Edge)을 긋습니다.
        for full_path, rel_path in file_paths:
            # 이 파일이 import 하고 있는 모듈 목록을 가져옵니다.
            imported_modules = self._parse_imports(full_path)

            for module in imported_modules:
                # import 된 모듈 이름(예: utils.helper)을 파일 경로(utlts/helper.py)로 추측해봅니다.
                potential_path = module.replace(".", "/") + ".py"

                # 만약 그 추측한 경로가 우리 그래프에 이미 있는 파일이라면
                if potential_path in self.graph.nodes:
                    # 두 파일 사이에 선(Edge)을 긋습니다.
                    self.graph.add_edge(rel_path, potential_path, type="import")

    def get_related_files(self, file_path, limit=3):
        """
        특정 파일과 연결된(관련 깊은) 파일들을 찾아줍니다.
        """
        if file_path not in self.graph:
            return []

        # 나와 연결된 파일들을 가져옵니다.
        # neighbors: 내가 import 한 파일들 + 나를 import 한 파일들
        related = list(self.graph.successors(file_path)) + list(
            self.graph.predecessors(file_path)
        )

        # 너무 많으면 limit 개수만큼만 자릅니다.
        return list(set(related))[:limit]

    def save(self, save_path):
        """
        만들어진 그래프 지도를 파일로 저장합니다.
        """
        with open(save_path, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self, load_path):
        """
        저장된 그래프 지도를 불러옵니다.
        """
        if os.path.exists(load_path):
            with open(load_path, "rb") as f:
                self.graph = pickle.load(f)
            return True
        return False
