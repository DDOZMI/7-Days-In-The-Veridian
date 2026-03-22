import json
from typing import Dict, List, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from models import Entity
from config import config
import tiktoken


class EntityMemory:
    """엔티티 기반 메모리 관리"""
    
    def __init__(self, llm: ChatGoogleGenerativeAI, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Entity 추출 프롬프트 로드
        try:
            with open('prompts/entity_extraction_prompt.txt', 'r', encoding='utf-8') as f:
                self.extraction_prompt = f.read()
        except FileNotFoundError:
            # 기본 프롬프트 제공
            self.extraction_prompt = self._get_default_extraction_prompt()
    
    def _get_default_extraction_prompt(self) -> str:
        """기본 엔티티 추출 프롬프트"""
        return """
Extract key entities from this conversation segment for a roleplay novel.

# Entity Types
- CHARACTER: People or beings with names, personalities, appearances
- LOCATION: Places, settings, environments
- OBJECT: Important items, artifacts, tools
- RELATIONSHIP: Connections between characters

# Output Format (JSON)
{
  "entities": [
    {
      "name": "entity name",
      "type": "CHARACTER|LOCATION|OBJECT|RELATIONSHIP",
      "attributes": {
        "key1": "value1",
        "key2": "value2"
      }
    }
  ]
}

# Guidelines
- Extract only EXPLICITLY mentioned entities
- For CHARACTER: include traits, appearance, personality, status
- For LOCATION: include description, atmosphere
- For RELATIONSHIP: include relationship_type and entities involved
- Use the same language as the conversation
- Keep attributes concise but informative

Conversation:
{conversation}

Extracted Entities (JSON only):
"""
    
    def count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        return len(self.encoding.encode(text))
    
    async def extract_entities(
        self, 
        messages: List[HumanMessage | AIMessage],
        session_id: str
    ) -> List[Entity]:
        """대화에서 엔티티 추출"""
        try:
            # 대화를 텍스트로 변환
            conversation_text = self._format_messages(messages)
            
            # 프롬프트 생성
            prompt = ChatPromptTemplate.from_template(self.extraction_prompt)
            chain = prompt | self.llm
            
            # 엔티티 추출
            response = await chain.ainvoke({"conversation": conversation_text})
            
            # JSON 파싱
            entities_data = self._parse_entity_response(response.content)
            
            # Entity 객체로 변환
            entities = []
            for entity_dict in entities_data.get("entities", []):
                entity = Entity(
                    name=entity_dict.get("name", "Unknown"),
                    type=entity_dict.get("type", "OBJECT").lower(),
                    attributes=entity_dict.get("attributes", {}),
                    session_id=session_id,
                    last_updated=datetime.now()
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            print(f"엔티티 추출 중 오류: {e}")
            return []
    
    def _format_messages(self, messages: List[HumanMessage | AIMessage]) -> str:
        """메시지를 텍스트로 포맷"""
        text = ""
        for msg in messages:
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            text += f"{role}: {msg.content}\n\n"
        return text
    
    def _parse_entity_response(self, response: str) -> Dict:
        """LLM 응답에서 JSON 추출"""
        try:
            # JSON 블록 찾기
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            
            # JSON 형식이 아니면 빈 딕셔너리 반환
            return {"entities": []}
            
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return {"entities": []}
    
    async def update_entities(
        self,
        new_entities: List[Entity],
        session_id: str
    ) -> Dict[str, Entity]:
        """엔티티 업데이트 (병합 및 저장)"""
        try:
            # 기존 엔티티 로드
            existing_entities = await self.load_entities(session_id)
            
            # 새 엔티티 병합
            for new_entity in new_entities:
                key = f"{new_entity.name}:{new_entity.type}"
                
                if key in existing_entities:
                    # 기존 엔티티 업데이트
                    existing = existing_entities[key]
                    existing.attributes.update(new_entity.attributes)
                    existing.last_updated = datetime.now()
                else:
                    # 새 엔티티 추가
                    existing_entities[key] = new_entity
            
            # Vector Store에 저장
            await self._save_entities_to_vectorstore(existing_entities, session_id)
            
            return existing_entities
            
        except Exception as e:
            print(f"엔티티 업데이트 중 오류: {e}")
            return {}
    
    async def load_entities(self, session_id: str) -> Dict[str, Entity]:
        """세션의 모든 엔티티 로드"""
        try:
            collection = self.vectorstore._collection
            results = collection.get(
                where={
                    "session_id": session_id,
                    "type": "entity"
                }
            )
            
            entities = {}
            if results and results['documents']:
                for doc, metadata in zip(results['documents'], results['metadatas']):
                    entity_data = json.loads(doc)
                    entity = Entity(**entity_data)
                    key = f"{entity.name}:{entity.type}"
                    entities[key] = entity
            
            return entities
            
        except Exception as e:
            print(f"엔티티 로드 중 오류: {e}")
            return {}
    
    async def _save_entities_to_vectorstore(
        self,
        entities: Dict[str, Entity],
        session_id: str
    ):
        """엔티티를 Vector Store에 저장"""
        try:
            # 기존 엔티티 삭제
            collection = self.vectorstore._collection
            old_results = collection.get(
                where={
                    "session_id": session_id,
                    "type": "entity"
                }
            )
            if old_results and old_results['ids']:
                collection.delete(ids=old_results['ids'])
            
            # 새 엔티티 저장
            if entities:
                texts = []
                metadatas = []
                
                for key, entity in entities.items():
                    # 엔티티를 JSON 문자열로 저장
                    entity_json = entity.model_dump_json()
                    texts.append(entity_json)
                    
                    metadata = {
                        "session_id": session_id,
                        "type": "entity",
                        "entity_type": entity.type,
                        "entity_name": entity.name,
                        "timestamp": entity.last_updated.isoformat()
                    }
                    metadatas.append(metadata)
                
                self.vectorstore.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )
                
                print(f"엔티티 저장 완료: {len(entities)}개 (세션: {session_id})")
                
        except Exception as e:
            print(f"엔티티 저장 중 오류: {e}")
    
    def get_entity_context(
        self,
        entities: Dict[str, Entity],
        max_tokens: int = None
    ) -> str:
        """엔티티를 프롬프트 컨텍스트로 변환"""
        if not entities:
            return ""
        
        if max_tokens is None:
            max_tokens = config.memory.entity_max_tokens
        
        context = "=== 주요 등장인물 및 설정 ===\n\n"
        
        # 타입별로 그룹화
        characters = []
        locations = []
        relationships = []
        others = []
        
        for entity in entities.values():
            if entity.type == "character":
                characters.append(entity)
            elif entity.type == "location":
                locations.append(entity)
            elif entity.type == "relationship":
                relationships.append(entity)
            else:
                others.append(entity)
        
        # 섹션별 추가
        sections = []
        
        if characters:
            section = "[Character]\n" + "\n".join([e.to_context_string() for e in characters])
            sections.append(section)
        
        if locations:
            section = "[Location]\n" + "\n".join([e.to_context_string() for e in locations])
            sections.append(section)
        
        if relationships:
            section = "[Relationship]\n" + "\n".join([e.to_context_string() for e in relationships])
            sections.append(section)
        
        if others:
            section = "[Others]\n" + "\n".join([e.to_context_string() for e in others])
            sections.append(section)
        
        full_context = context + "\n\n".join(sections)
        
        # 토큰 제한 적용
        if self.count_tokens(full_context) > max_tokens:
            # 우선순위: 인물 > 관계 > 장소 > 기타
            truncated_sections = []
            current_tokens = self.count_tokens(context)
            
            for section in sections:
                section_tokens = self.count_tokens(section)
                if current_tokens + section_tokens <= max_tokens:
                    truncated_sections.append(section)
                    current_tokens += section_tokens
                else:
                    break
            
            full_context = context + "\n\n".join(truncated_sections)
        
        return full_context
    
    async def clear_entities(self, session_id: str):
        """세션의 모든 엔티티 삭제"""
        try:
            collection = self.vectorstore._collection
            results = collection.get(
                where={
                    "session_id": session_id,
                    "type": "entity"
                }
            )
            
            if results and results['ids']:
                collection.delete(ids=results['ids'])
                print(f"엔티티 삭제 완료: {len(results['ids'])}개 (세션: {session_id})")
                
        except Exception as e:
            print(f"엔티티 삭제 중 오류: {e}")