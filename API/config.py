from dataclasses import dataclass, field
from typing import Optional, List
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


@dataclass
class MemoryConfig:
    """메모리 관리 설정"""
    # 요약 임계값
    summarize_threshold: int = 15000
    
    # 검색 설정
    memory_top_k: int = 5
    recent_messages_keep: int = 12
    
    # 메모리 비율 설정
    memory_context_ratio: float = 0.20
    short_term_memory_ratio: float = 0.15
    
    # 토큰 제한
    min_memory_context_tokens: int = 2000
    max_memory_context_tokens: int = 50000
    
    # Entity Memory 설정
    entity_enabled: bool = True
    entity_max_tokens: int = 3000
    entity_update_frequency: int = 2  # N개 메시지마다 업데이트


@dataclass
class ModelConfig:
    """LLM 모델 설정"""
    # 메인 대화 모델
    main_model: str = "gemini-2.5-pro-preview-06-05"
    main_temperature: float = 1.6
    main_top_p: float = 0.9
    main_top_k: int = 50
    
    # 요약 모델
    summary_model: str = "gemini-2.5-flash-preview-05-20"
    summary_temperature: float = 0.3
    
    # Entity 추출 모델
    entity_model: str = "gemini-2.5-flash-preview-05-20"
    entity_temperature: float = 0.2
    
    # Embedding 모델
    embedding_model: str = "models/embedding-001"


@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    chroma_persist_dir: str = "./chroma_db"
    conversation_collection: str = "conversation_memory"
    entity_collection: str = "entity_memory"


@dataclass
class APIConfig:
    """API 서버 설정"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class AppConfig:
    """전체 애플리케이션 설정"""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    google_api_key: Optional[str] = None
    
    def __post_init__(self):
        """환경변수에서 설정 값 오버라이드"""
        # API Key
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Memory 설정
        if val := os.getenv("SUMMARIZE_THRESHOLD"):
            self.memory.summarize_threshold = int(val)
        if val := os.getenv("MEMORY_TOP_K"):
            self.memory.memory_top_k = int(val)
        if val := os.getenv("RECENT_MESSAGES_KEEP"):
            self.memory.recent_messages_keep = int(val)
        if val := os.getenv("MEMORY_CONTEXT_RATIO"):
            self.memory.memory_context_ratio = float(val)
        if val := os.getenv("SHORT_TERM_MEMORY_RATIO"):
            self.memory.short_term_memory_ratio = float(val)
        if val := os.getenv("MIN_MEMORY_TOKENS"):
            self.memory.min_memory_context_tokens = int(val)
        if val := os.getenv("MAX_MEMORY_TOKENS"):
            self.memory.max_memory_context_tokens = int(val)
        if val := os.getenv("ENTITY_MEMORY_ENABLED"):
            self.memory.entity_enabled = val.lower() == "true"
        if val := os.getenv("ENTITY_MAX_TOKENS"):
            self.memory.entity_max_tokens = int(val)
        if val := os.getenv("ENTITY_UPDATE_FREQ"):
            self.memory.entity_update_frequency = int(val)
        
        # Model 설정
        if val := os.getenv("MAIN_MODEL"):
            self.model.main_model = val
        if val := os.getenv("MAIN_TEMP"):
            self.model.main_temperature = float(val)
        if val := os.getenv("MAIN_TOP_P"):
            self.model.main_top_p = float(val)
        if val := os.getenv("MAIN_TOP_K"):
            self.model.main_top_k = int(val)
        if val := os.getenv("SUMMARY_MODEL"):
            self.model.summary_model = val
        if val := os.getenv("SUMMARY_TEMP"):
            self.model.summary_temperature = float(val)
        if val := os.getenv("ENTITY_MODEL"):
            self.model.entity_model = val
        if val := os.getenv("ENTITY_TEMP"):
            self.model.entity_temperature = float(val)
        if val := os.getenv("EMBEDDING_MODEL"):
            self.model.embedding_model = val
        
        # Database 설정
        if val := os.getenv("CHROMA_DIR"):
            self.database.chroma_persist_dir = val
        
        # API 설정
        if val := os.getenv("API_HOST"):
            self.api.host = val
        if val := os.getenv("API_PORT"):
            self.api.port = int(val)
        if val := os.getenv("DEBUG"):
            self.api.debug = val.lower() == "true"
        if val := os.getenv("CORS_ORIGINS"):
            self.api.cors_origins = [origin.strip() for origin in val.split(",")]


def get_config() -> AppConfig:
    """설정 인스턴스 생성 및 반환"""
    return AppConfig()


# 전역 설정 인스턴스 (싱글톤 패턴)
_config_instance = None


def load_config() -> AppConfig:
    """설정 로드 (캐싱)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = get_config()
    return _config_instance


# 기본 export
config = load_config()