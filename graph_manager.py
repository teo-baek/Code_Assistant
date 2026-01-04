# Neo4j DB를 사용해서 코드 간의 복잡한 연결 고리를 관리하는 파일

from neo4j import GraphDatabase

class CodeGraphManager:
    """
    코드 파일들이 서로 어떻게 호출되고 참조하는지 선으로 연결해 관리하는 클래스
    """
    def __init__(self, uri, user, password):
        # Neo4j DB에 접속하기 위한 주소와 아이디, 암호를 설정
        self.driver = GraphDatabase.driver(uri, auth= (user, password))

    def close(self):
        self.driver.close()

    def add_relation(self, source_file, target_file, rel_type= "IMPORT"):
        """
        파일 A가 파일 B를 참조한다는 연결 고리를 DB에 기록
        """
        with self.driver.session() as session:
            # Cypher라는 그래프 전용 언어를 사용해 데이터를 저장함. (MERGE: 없으면 만들고, 있으면 유지함)
            session.run("""
                MERGE (a:File {name: $source})
                MERGE (b:File {name: $target})
                MERGE (a)-[r:REL {type: $rel}]->(b)            
            """, source=source_file, target= target_file, rel= rel_type)
    
    def get_related_nodes(self, file_name):
        """
        특정 파일과 연결된 모든 이웃 파일들의 이름을 찾아옴.
        """
        with self.driver.session() as session:
            # 이 파일과 연결된 모든 점(Node)들을 찾아오는 쿼리를 실행함.
            result = session.run("""
                MATCH (a:File  {name: $name})-[r]-(neighbor)
                RETURN neighbor.name as name
            """, name= file_name)
            # 결과물에서 이름만 리스트에 담아 돌려줌
            return [record["name"] for record in result]