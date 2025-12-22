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