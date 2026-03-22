from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """채팅 메시지 모델"""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="메시지 내용")


class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str = Field(..., description="사용자 입력 메시지")
    history: List[ChatMessage] = Field(default_factory=list, description="대화 이력")
    session_id: str = Field(default="default_session", description="세션 ID")


class ChatChunkResponse(BaseModel):
    """스트리밍 응답 청크"""
    chunk: Optional[str] = None
    error: Optional[str] = None


class ChatCompleteResponse(BaseModel):
    """채팅 완료 응답"""
    done: bool = True
    full_response: str
    summarized: bool
    entity_updated: bool
    short_term_size: int
    total_history_size: int
    total_tokens: int
    memory_allocation: int
    short_term_allocation: int


class ClearMemoryRequest(BaseModel):
    """메모리 삭제 요청"""
    session_id: Optional[str] = None


class MemoryStatsResponse(BaseModel):
    """메모리 통계 응답"""
    session_id: Optional[str] = None
    conversation_memory_count: int = 0
    entity_count: int = 0
    total_memories: Optional[int] = None
    sessions: Optional[Dict[str, int]] = None
    entities: Optional[Dict[str, Any]] = None


class Entity(BaseModel):
    """엔티티 모델"""
    name: str
    type: str  # 'character', 'location', 'object', 'relationship'
    attributes: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)
    session_id: str
    
    def to_context_string(self) -> str:
        """프롬프트에 포함할 문자열 형식으로 변환"""
        if self.type == "character":
            attrs = ", ".join([f"{k}: {v}" for k, v in self.attributes.items()])
            return f"• {self.name} (인물) - {attrs}"
        elif self.type == "location":
            desc = self.attributes.get("description", "")
            return f"• {self.name} (장소) - {desc}"
        elif self.type == "relationship":
            rel_type = self.attributes.get("relationship_type", "")
            entities = self.attributes.get("entities", [])
            if len(entities) >= 2:
                return f"• 관계: {entities[0]} ↔ {entities[1]} ({rel_type})"
        else:
            attrs = ", ".join([f"{k}: {v}" for k, v in self.attributes.items()])
            return f"• {self.name} ({self.type}) - {attrs}"
        return f"• {self.name}"