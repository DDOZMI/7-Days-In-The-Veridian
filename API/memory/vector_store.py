from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import config


class VectorStoreManager:
    """Vector Store 관리 클래스"""
    
    def __init__(self, embeddings: GoogleGenerativeAIEmbeddings):
        self.embeddings = embeddings
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Vector Store 초기화"""
        try:
            self.vectorstore = Chroma(
                persist_directory=config.database.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name=config.database.conversation_collection
            )
            print("✓ Vector Store 초기화 완료")
            
        except Exception as e:
            print(f"✗ Vector Store 초기화 중 오류: {e}")
            raise
    
    def get_vectorstore(self) -> Chroma:
        """Vector Store 인스턴스 반환"""
        return self.vectorstore
    
    def reset_collection(self):
        """컬렉션 재설정"""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
            
            self.vectorstore = Chroma(
                persist_directory=config.database.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name=config.database.conversation_collection
            )
            
            print("✓ Vector Store 컬렉션 재설정 완료")
            
        except Exception as e:
            print(f"✗ 컬렉션 재설정 중 오류: {e}")
            raise