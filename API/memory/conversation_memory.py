from typing import List, Tuple
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import tiktoken

from config import config


class ConversationMemory:
    """대화 메모리 관리 클래스"""
    
    def __init__(
        self,
        summarize_llm: ChatGoogleGenerativeAI,
        vectorstore
    ):
        self.summarize_llm = summarize_llm
        self.vectorstore = vectorstore
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # 요약 프롬프트 로드
        self.summarize_prompt = self._load_summarize_prompt()
    
    def _load_summarize_prompt(self) -> str:
        """요약 프롬프트 로드"""
        # 기존 코드의 SUMMARIZE_PROMPT 사용
        return """
Summarize conversations according to following instructions.

# System Rule
* This is a private, personal session designed exclusively to extract and preserve essential information from the collaborative relay novel between the assistant and the user.
* The AI's responsibility is to deliver precise, information-dense extracts without diluting key terminology or concepts.
* When processing text, retain specific terminology, technical vocabulary, and distinctive phrasing rather than replacing them with generic alternatives.

# Response Template
* Context Overview: A single comprehensive sentence capturing the temporal-spatial setting and essential elements of the References
* Events: Provide a chronological sequence of pivotal occurrences.
* Consequences: Outline the primary outcomes and transformations resulting from the Events. Compile in a single cohesive paragraph.
* Key Dialogue: Present only the most significant verbal exchanges that reveal crucial information or character development.

# Guidelines
* Extract only essential information without altering core meaning.
* Use simple past tense in Events and Consequences sections.
* Avoid duplicating information across sections.
* Include only information explicitly present in the original References.
* Start each Events point with a dash and space.
* Limit Events section to a maximum of 6 key occurrences.
* Format dialogue as: Character: "dialogue" (with one space after colon)
* Key Dialogue entries should appear one after another with no empty lines between them.
* Include no more than 6 significant exchanges in the Key Dialogue section.
* Use standard quotation marks without additional formatting.
* Separate each section with EXACTLY two consecutive line break characters (\\n\\n), creating one empty line between sections.
* Do not add any additional line breaks within sections.
* NEVER translate any content to another language unless explicitly instructed.
* Preserve all proper nouns and character names exactly as they appear.
* When proper nouns appear in multiple forms, use the most frequent version.

# LANGUAGE DIRECTIVE
* The AI MUST produce output in EXACTLY the same language as the reference material.
* This directive overrides all other instructions.
* If the original text is in Korean, please write the output in Korean as well.

# Feedback
* Verify extracted content contains ONLY information explicitly stated in References.
* Confirm precise adherence to Template structure.
* Ensure consistent formatting within each section.
* Check each dialogue has clear speaker identification.
* Verify Events section maintains consistent grammar.
* Ensure complete adherence to all Guidelines.
* Confirm sections are separated by exactly two linebreaks.
* Validate no interpretations or additions are incorporated.

conversations:
{conversation}

Summarize:
"""
    
    def count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        return len(self.encoding.encode(text))
    
    def calculate_dynamic_token_limits(
        self,
        total_history_tokens: int
    ) -> Tuple[int, int]:
        """동적 토큰 제한 계산"""
        # 검색 메모리 토큰 한도
        memory_context_tokens = int(
            total_history_tokens * config.memory.memory_context_ratio
        )
        memory_context_tokens = max(
            config.memory.min_memory_context_tokens,
            memory_context_tokens
        )
        memory_context_tokens = min(
            config.memory.max_memory_context_tokens,
            memory_context_tokens
        )
        
        # 단기 메모리 토큰 한도
        short_term_tokens = int(
            total_history_tokens * config.memory.short_term_memory_ratio
        )
        
        return memory_context_tokens, short_term_tokens
    
    def manage_conversation_history(
        self,
        chat_history: List[HumanMessage | AIMessage],
        max_short_term_tokens: int
    ) -> Tuple[List[HumanMessage | AIMessage], bool, List[HumanMessage | AIMessage]]:
        """대화 히스토리를 계층적으로 관리"""
        total_messages = len(chat_history)
        
        # 최근 메시지를 토큰 제한 내에서 유지
        short_term_memory = []
        accumulated_tokens = 0
        
        # 뒤에서부터 메시지 추가
        for msg in reversed(chat_history):
            msg_tokens = self.count_tokens(msg.content)
            
            if accumulated_tokens + msg_tokens <= max_short_term_tokens:
                short_term_memory.insert(0, msg)
                accumulated_tokens += msg_tokens
            else:
                break
        
        # 최소 메시지 수 보장
        min_keep = config.memory.recent_messages_keep
        if len(short_term_memory) < min_keep and len(short_term_memory) < total_messages:
            remaining = min(
                min_keep - len(short_term_memory),
                total_messages - len(short_term_memory)
            )
            for i in range(remaining):
                idx = len(short_term_memory)
                if idx < total_messages:
                    short_term_memory.insert(0, chat_history[total_messages - idx - 1])
        
        # 요약 대상 메시지
        messages_to_summarize = chat_history[:len(chat_history) - len(short_term_memory)]
        should_summarize = len(messages_to_summarize) > 0
        
        print(
            f"전체 메시지: {total_messages}, "
            f"단기 기억: {len(short_term_memory)} ({accumulated_tokens} 토큰), "
            f"요약 대상: {len(messages_to_summarize)}"
        )
        
        return short_term_memory, should_summarize, messages_to_summarize
    
    async def summarize_conversation(
        self,
        messages: List[HumanMessage | AIMessage]
    ) -> str:
        """대화 요약"""
        try:
            conversation_text = self._format_conversation(messages)
            
            prompt = ChatPromptTemplate.from_template(self.summarize_prompt)
            chain = prompt | self.summarize_llm
            
            response = await chain.ainvoke({"conversation": conversation_text})
            return response.content
            
        except Exception as e:
            print(f"요약 중 오류: {e}")
            return ""
    
    def _format_conversation(
        self,
        messages: List[HumanMessage | AIMessage]
    ) -> str:
        """대화를 요약용 텍스트로 포맷"""
        text = ""
        for msg in messages:
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            text += f"{role}: {msg.content}\n\n"
        return text
    
    async def save_summary_to_vectorstore(
        self,
        summary: str,
        session_id: str
    ):
        """요약을 Vector Store에 저장"""
        try:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "type": "conversation_summary"
            }
            
            self.vectorstore.add_texts(
                texts=[summary],
                metadatas=[metadata]
            )
            
            print(f"메모리 저장 완료: {session_id}")
            
        except Exception as e:
            print(f"Vector Store 저장 중 오류: {e}")
    
    def retrieve_relevant_memories(
        self,
        query: str,
        session_id: str,
        max_tokens: int
    ) -> str:
        """관련 과거 메모리 검색"""
        try:
            if self.vectorstore is None:
                return ""
            
            # 세션별 필터링을 위한 검색
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=config.memory.memory_top_k * 2,
                filter={
                    "session_id": session_id,
                    "type": "conversation_summary"
                }
            )
            
            if not results:
                return ""
            
            # Relevance score 기반 정렬
            results.sort(key=lambda x: x[1])
            
            # 동적 토큰 제한 내에서 메모리 구성
            memories = []
            total_tokens = 0
            
            for i, (doc, score) in enumerate(results[:config.memory.memory_top_k], 1):
                memory_text = f"[과거 대화 {i}]\n{doc.page_content}"
                memory_tokens = self.count_tokens(memory_text)
                
                if total_tokens + memory_tokens > max_tokens:
                    remaining_tokens = max_tokens - total_tokens
                    if remaining_tokens > 100:
                        truncated = self._truncate_to_token_limit(
                            doc.page_content,
                            remaining_tokens - 20
                        )
                        memories.append(f"[과거 대화 {i}]\n{truncated}...")
                    break
                
                memories.append(memory_text)
                total_tokens += memory_tokens
            
            result = "\n\n".join(memories)
            print(f"검색된 메모리 토큰 수: {total_tokens} / 할당량: {max_tokens}")
            return result
            
        except Exception as e:
            print(f"메모리 검색 중 오류: {e}")
            return ""
    
    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """텍스트를 지정된 토큰 수로 자르기"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    async def clear_conversation_memory(self, session_id: str = None):
        """대화 메모리 삭제"""
        try:
            collection = self.vectorstore._collection
            
            if session_id:
                results = collection.get(
                    where={
                        "session_id": session_id,
                        "type": "conversation_summary"
                    }
                )
                
                if results and results['ids']:
                    collection.delete(ids=results['ids'])
                    print(f"세션 {session_id}의 {len(results['ids'])}개 메모리 삭제")
                    return len(results['ids'])
            else:
                # 전체 대화 메모리 삭제
                results = collection.get(
                    where={"type": "conversation_summary"}
                )
                if results and results['ids']:
                    collection.delete(ids=results['ids'])
                    print(f"전체 대화 메모리 삭제: {len(results['ids'])}개")
                    return len(results['ids'])
            
            return 0
            
        except Exception as e:
            print(f"메모리 삭제 중 오류: {e}")
            return 0