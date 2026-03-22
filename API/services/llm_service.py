from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any

from config import config


class LLMService:
    """LLM 서비스 클래스"""
    
    def __init__(self):
        self.main_llm = None
        self.summarize_llm = None
        self.entity_llm = None
        self.embeddings = None
        
        self._initialize_models()
        self._load_prompts()
    
    def _initialize_models(self):
        """모델 초기화"""
        try:
            # 메인 대화 LLM
            self.main_llm = ChatGoogleGenerativeAI(
                model=config.model.main_model,
                temperature=config.model.main_temperature,
                top_p=config.model.main_top_p,
                top_k=config.model.main_top_k,
                google_api_key=config.google_api_key,
                streaming=True
            )
            
            # 요약 LLM
            self.summarize_llm = ChatGoogleGenerativeAI(
                model=config.model.summary_model,
                temperature=config.model.summary_temperature,
                google_api_key=config.google_api_key,
                streaming=False
            )
            
            # Entity 추출 LLM
            self.entity_llm = ChatGoogleGenerativeAI(
                model=config.model.entity_model,
                temperature=config.model.entity_temperature,
                google_api_key=config.google_api_key,
                streaming=False
            )
            
            # Embeddings 모델
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=config.model.embedding_model,
                google_api_key=config.google_api_key
            )
            
            print("✓ LLM 모델 초기화 완료")
            
        except Exception as e:
            print(f"✗ 모델 초기화 중 오류: {e}")
            raise
    
    def _load_prompts(self):
        """프롬프트 파일 로드"""
        try:
            with open('prompts/base_prompt.txt', 'r', encoding='utf-8') as f:
                self.base_prompt = f.read()
            
            with open('prompts/worldview_prompt.txt', 'r', encoding='utf-8') as f:
                self.worldview_prompt = f.read()
            
            with open('prompts/guideline_prompt.txt', 'r', encoding='utf-8') as f:
                self.guideline_prompt = f.read()
            
            print("✓ 프롬프트 파일 로드 완료")
            
        except FileNotFoundError as e:
            print(f"✗ 프롬프트 파일을 찾을 수 없습니다: {e}")
            # 기본 프롬프트 설정
            self.base_prompt = "You are a helpful AI assistant for roleplay novels."
            self.worldview_prompt = ""
            self.guideline_prompt = ""
    
    def create_enhanced_prompt(self) -> ChatPromptTemplate:
        """향상된 프롬프트 템플릿 생성"""
        return ChatPromptTemplate.from_messages([
            ("system", self.base_prompt),
            ("system", self.worldview_prompt),
            ("system", self.guideline_prompt),
            ("system", """
# Entity Context (Major character & narrative settings)
{entity_context}

# Past Conversation Context
Following texts is relevant to the current question from a past conversation.
Refer to this to provide a more contextual response.
If there's no past conversation, ignore it and focus on the current conversation.

{memory_context}
"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ])
    
    def create_chain_with_memory(
        self,
        get_memory_context_func,
        get_entity_context_func
    ):
        """메모리 검색이 통합된 체인 생성"""
        def prepare_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """메모리와 엔티티 컨텍스트 준비"""
            user_input = inputs.get("user_input", "")
            
            # 메모리 검색
            memory_context = get_memory_context_func(user_input)
            
            # 엔티티 컨텍스트
            entity_context = get_entity_context_func()
            
            return {
                **inputs,
                "memory_context": memory_context if memory_context else "no memory context",
                "entity_context": entity_context if entity_context else "no entity context"
            }
        
        enhanced_prompt = self.create_enhanced_prompt()
        
        chain = (
            RunnableLambda(prepare_context)
            | enhanced_prompt
            | self.main_llm
        )
        
        return chain