import json
import asyncio
from typing import Dict
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from config import config
from models import (
    ChatRequest, ChatChunkResponse, ChatCompleteResponse,
    ClearMemoryRequest, MemoryStatsResponse
)
from services.llm_service import LLMService
from memory.vector_store import VectorStoreManager
from memory.conversation_memory import ConversationMemory
from memory.entity_memory import EntityMemory


# FastAPI 앱 초기화
app = FastAPI(
    title="Roleplay Chatbot API",
    description="Role-play chat api generator",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 서비스 인스턴스
llm_service: LLMService = None
vector_store_manager: VectorStoreManager = None
conversation_memory: ConversationMemory = None
entity_memory: EntityMemory = None

# 세션별 엔티티 캐시
entity_cache: Dict[str, Dict] = {}


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 초기화"""
    global llm_service, vector_store_manager, conversation_memory, entity_memory
    
    print("=" * 50)
    print("FastAPI Roleplay Chatbot Server Starting...")
    print("=" * 50)
    
    try:
        # LLM 서비스 초기화
        llm_service = LLMService()
        
        # Vector Store 초기화
        vector_store_manager = VectorStoreManager(llm_service.embeddings)
        vectorstore = vector_store_manager.get_vectorstore()
        
        # Conversation Memory 초기화
        conversation_memory = ConversationMemory(
            summarize_llm=llm_service.summarize_llm,
            vectorstore=vectorstore
        )
        
        # Entity Memory 초기화
        entity_memory = EntityMemory(
            llm=llm_service.entity_llm,
            vectorstore=vectorstore
        )
        
        print("=" * 50)
        print("모든 서비스 초기화 완료")
        print(f"서버 주소: http://{config.api.host}:{config.api.port}")
        print(f"Entity Memory: {'활성화' if config.memory.entity_enabled else '비활성화'}")
        print("=" * 50)
        
    except Exception as e:
        print(f"초기화 실패: {e}")
        raise


@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "running",
        "version": "2.0.0",
        "entity_memory": config.memory.entity_enabled
    }


@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """채팅 엔드포인트 (스트리밍)"""
    if llm_service is None or llm_service.main_llm is None:
        raise HTTPException(status_code=500, detail="LLM 서비스가 초기화되지 않았습니다")
    
    user_input = request.message
    history_data = request.history
    session_id = request.session_id
    
    if not user_input:
        raise HTTPException(status_code=400, detail="메시지가 입력되지 않았습니다")
    
    # 채팅 히스토리 파싱
    chat_history = []
    total_history_tokens = 0
    
    for msg in history_data:
        content = msg.content
        total_history_tokens += conversation_memory.count_tokens(content)
        
        if msg.role == 'user':
            chat_history.append(HumanMessage(content=content))
        elif msg.role == 'assistant':
            chat_history.append(AIMessage(content=content))
    
    # 동적 토큰 한도 계산
    max_memory_tokens, max_short_term_tokens = \
        conversation_memory.calculate_dynamic_token_limits(total_history_tokens)
    
    print(f"\n{'='*60}")
    print(f"새 메시지 수신 (세션: {session_id})")
    print(f"총 히스토리 토큰: {total_history_tokens}")
    print(f"메모리 할당: {max_memory_tokens} ({config.memory.memory_context_ratio*100}%)")
    print(f"단기 메모리 할당: {max_short_term_tokens} ({config.memory.short_term_memory_ratio*100}%)")
    print(f"{'='*60}\n")
    
    # 계층적 메모리 관리
    short_term_memory, should_summarize, messages_to_summarize = \
        conversation_memory.manage_conversation_history(
            chat_history,
            max_short_term_tokens
        )
    
    # Entity 업데이트 필요 여부 판단
    entity_updated = False
    should_update_entity = (
        config.memory.entity_enabled and
        len(chat_history) % config.memory.entity_update_frequency == 0
    )
    
    # 백그라운드 작업: 요약 및 Entity 추출
    if should_summarize and messages_to_summarize:
        background_tasks.add_task(
            background_summarize,
            messages_to_summarize,
            session_id
        )
    
    if should_update_entity and len(chat_history) >= 2:
        # 최근 N개 메시지에서 Entity 추출
        recent_for_entity = chat_history[-4:] if len(chat_history) >= 4 else chat_history
        background_tasks.add_task(
            background_extract_entities,
            recent_for_entity,
            session_id
        )
        entity_updated = True
    
    # 세션의 엔티티 로드 (캐시 활용)
    if session_id not in entity_cache:
        entity_cache[session_id] = await entity_memory.load_entities(session_id)
    
    current_entities = entity_cache[session_id]
    
    # 체인 생성
    def get_memory_context(user_input: str) -> str:
        return conversation_memory.retrieve_relevant_memories(
            user_input,
            session_id,
            max_memory_tokens
        )
    
    def get_entity_context() -> str:
        return entity_memory.get_entity_context(current_entities)
    
    chain = llm_service.create_chain_with_memory(
        get_memory_context,
        get_entity_context
    )
    
    # 스트리밍 응답 생성
    async def generate():
        try:
            full_response = ""
            
            async for chunk in chain.astream({
                "user_input": user_input,
                "history": short_term_memory
            }):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    
                    # 청크 전송
                    chunk_data = ChatChunkResponse(chunk=chunk.content)
                    yield f"data: {chunk_data.model_dump_json()}\n\n"
            
            # 완료 신호
            completion = ChatCompleteResponse(
                done=True,
                full_response=full_response,
                summarized=should_summarize,
                entity_updated=entity_updated,
                short_term_size=len(short_term_memory),
                total_history_size=len(chat_history),
                total_tokens=total_history_tokens,
                memory_allocation=max_memory_tokens,
                short_term_allocation=max_short_term_tokens
            )
            
            yield f"data: {completion.model_dump_json()}\n\n"
            
        except Exception as e:
            error_data = ChatChunkResponse(error=str(e))
            yield f"data: {error_data.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


async def background_summarize(
    messages,
    session_id: str
):
    """백그라운드 요약 작업"""
    try:
        print(f"백그라운드 요약 시작 (세션: {session_id}, 메시지: {len(messages)}개)")
        
        summary = await conversation_memory.summarize_conversation(messages)
        
        if summary:
            await conversation_memory.save_summary_to_vectorstore(summary, session_id)
            print(f"요약 완료 (세션: {session_id})")
        
    except Exception as e:
        print(f"백그라운드 요약 오류: {e}")


async def background_extract_entities(
    messages,
    session_id: str
):
    """백그라운드 Entity 추출 작업"""
    try:
        print(f"백그라운드 Entity 추출 시작 (세션: {session_id})")
        
        # Entity 추출
        new_entities = await entity_memory.extract_entities(messages, session_id)
        
        if new_entities:
            # Entity 업데이트
            updated_entities = await entity_memory.update_entities(new_entities, session_id)
            
            # 캐시 업데이트
            entity_cache[session_id] = updated_entities
            
            print(f"Entity 추출 완료 (세션: {session_id}, {len(new_entities)}개 추출)")
        
    except Exception as e:
        print(f"백그라운드 Entity 추출 오류: {e}")


@app.post("/clear_memory")
async def clear_memory(request: ClearMemoryRequest):
    """메모리 삭제"""
    try:
        session_id = request.session_id
        
        if session_id:
            # 특정 세션 메모리 삭제
            conv_count = await conversation_memory.clear_conversation_memory(session_id)
            await entity_memory.clear_entities(session_id)
            
            # 캐시에서도 제거
            if session_id in entity_cache:
                del entity_cache[session_id]
            
            return {
                "message": f"Session {session_id} memory cleared",
                "conversation_deleted": conv_count,
                "entities_cleared": True
            }
        else:
            # 전체 메모리 삭제
            await conversation_memory.clear_conversation_memory()
            
            # 전체 Entity 삭제
            for sid in list(entity_cache.keys()):
                await entity_memory.clear_entities(sid)
            
            entity_cache.clear()
            
            # Vector Store 재설정
            vector_store_manager.reset_collection()
            
            return {
                "message": "All memory cleared",
                "full_reset": True
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory_stats")
async def memory_stats(session_id: str = None):
    """메모리 통계 조회"""
    try:
        vectorstore = vector_store_manager.get_vectorstore()
        collection = vectorstore._collection
        
        if session_id:
            # 특정 세션 통계
            conv_results = collection.get(
                where={
                    "session_id": session_id,
                    "type": "conversation_summary"
                }
            )
            conv_count = len(conv_results['ids']) if conv_results['ids'] else 0
            
            # Entity 통계
            entities = await entity_memory.load_entities(session_id)
            
            return MemoryStatsResponse(
                session_id=session_id,
                conversation_memory_count=conv_count,
                entity_count=len(entities),
                entities={k: v.model_dump() for k, v in entities.items()}
            )
        else:
            # 전체 통계
            all_results = collection.get()
            total_count = len(all_results['ids']) if all_results['ids'] else 0
            
            # 세션별 카운트
            session_counts = {}
            if all_results['metadatas']:
                for metadata in all_results['metadatas']:
                    sid = metadata.get('session_id', 'unknown')
                    session_counts[sid] = session_counts.get(sid, 0) + 1
            
            return MemoryStatsResponse(
                total_memories=total_count,
                sessions=session_counts
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug
    )