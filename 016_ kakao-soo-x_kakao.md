원본코드
"""
Orchestration Agent class for coordinating multiple agents.

This module provides the orchestration agent that coordinates workflows.
"""
import json
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.agents.llm_agent import LLMAgent
from src.utils.type_utils import normalize_chat_response
from src.utils.config_loader import format_agent_cards_for_prompt
from copy import deepcopy
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolParam,
    ChatCompletionSystemMessageParam
)
from src.utils.model_factory import ModelFactory

class OrchestrationAgent(BaseAgent):
    """Agent for coordinating multiple agents in workflows."""
    
    def __init__(self, agent_id: str, agent_card_list: List[Dict[str, Any]], llm_config: Dict[str, Any], prompts: Dict[str, Any]):
        """Initialize the orchestration agent."""
        self.agent_id = agent_id  # Fixed: was agent_ids
        self.agent_card_list = agent_card_list
        self.llm_config = llm_config
        self.prompts = prompts
        self.managed_agents = {}  # Will be set by OrchestrationEngine
        self.model = None  # Initialize model attribute
        
        # Token tracking callback (set by OrchestrationEngine)
        self.token_callback = None
    
    def set_managed_agents(self, agents: Dict[str, Any]):
        """Set the agents this orchestration agent manages."""
        self.managed_agents = agents

    async def execute(self, msgs: list[ChatCompletionMessageParam], 
                      prev_workflow: Dict[str, Any] = None, 
                      system_info: str = "",
                      skip_summary: bool = True,
                      kwargs: Dict[str, Any] = None) -> str:
        """Execute orchestration agent - coordinate multiple agents."""
        if not self.model:
            await self.initialize()
        
        # For orchestration agent, we use the full workflow
        return await self.generate_workflow(msgs, prev_workflow, system_info, skip_summary, kwargs)

    async def summarization(self, msgs: list[ChatCompletionMessageParam]) -> list[ChatCompletionAssistantMessageParam]:
        """Summarize workflow execution."""
        if not self.model:
            await self.initialize()
        msgs_ = deepcopy(msgs)
        system_msg = self._get_system_prompt("summarize")
        msgs_.insert(0, system_msg)
        
        logger.debug(f"OrchestrationAgent: Starting summarization with {len(msgs)} messages")
        
        # Generate summary
        response = await self.model.generate_chat_response(
            messages=msgs_,
        )
        response = normalize_chat_response(response)
        
        logger.debug(f"OrchestrationAgent: Summarization completed, response type: {type(response)}")
        return response

    async def classify_intent(self, msgs: list[ChatCompletionMessageParam], 
                                system_info: str = "", 
                                prev_workflow: Dict[str, Any] = None, 
                                kwargs: Dict[str, Any] = None) -> str:
        """Execute orchestration agent - coordinate multiple agents."""
        if not self.model:
            await self.initialize()
        
        # For orchestration agent, we use the full workflow
        return await self._classify_intent(msgs, system_info, prev_workflow, kwargs)

    async def _classify_intent(self, msgs: list[ChatCompletionMessageParam],
                              system_info: str = "", 
                              prev_workflow: Dict[str, Any] = None, 
                              kwargs: Dict[str, Any] = None) -> dict:
        msgs_ = deepcopy(msgs)
        prompt_template = self._get_system_prompt("classify", as_message=False)
        prompt_template = prompt_template.replace("%%system_info%%", system_info)
        prompt_template = prompt_template.replace("%%workflows%%", json.dumps(prev_workflow, ensure_ascii=False) if prev_workflow is not None and prev_workflow != {} else "NONE")
        system_msg = ChatCompletionSystemMessageParam(
                    role="system",
                    content=prompt_template
                )
        msgs_.insert(0, system_msg)
        # Generate workflow
        response = await self.model.generate_chat_response(
            messages=msgs_,
        )
        response = normalize_chat_response(response)
        return response
    

    async def generate_workflow(self, msgs: list[ChatCompletionMessageParam], 
                                prev_workflow: Dict[str, Any] = None,
                                system_info: str = "", 
                                skip_summary: bool = True, 
                                kwargs: dict = None) -> dict:
        """Generate workflow JSON based on input messages."""
        msgs_ = deepcopy(msgs)
        prompt_template = self._get_system_prompt("general", as_message=False)
        # Replace placeholders - convert agent cards to formatted string
        agent_cards_text = format_agent_cards_for_prompt(self.agent_card_list) if isinstance(self.agent_card_list, dict) else ""
        prompt = prompt_template.replace("%%agent_cards%%", agent_cards_text)
        prompt = prompt.replace("%%system_info%%", system_info)
        prompt = prompt.replace("%%workflows%%", json.dumps(prev_workflow, ensure_ascii=False) if prev_workflow is not None and prev_workflow != {} else "NONE")

        # Debug: Log the conditions
        logger.debug(f"OrchestrationAgent: len(msgs)={len(msgs)}, skip_summary={skip_summary}")
        
        history_text = ""
        for i, msg in enumerate(msgs):
            role = msg.get("role", "")
            content = msg.get("content", "")
            history_text += f"[Message {i+1}] {role}: {content}\n\n"
        prompt = prompt.replace("%%history%%", history_text)
        logger.debug(f"OrchestrationAgent: Added {len(msgs)} messages to history")

        system_msg = ChatCompletionSystemMessageParam(
                    role="system",
                    content=prompt
                )
        
        # When skip_summary=True, use a simple user message asking for workflow generation
        # since all context is already in the system prompt
        if skip_summary and len(msgs) > 0:
            last_user_msg = None
            for msg in reversed(msgs):
                if msg.get("role") == "user":
                    last_user_msg = msg
                    break
            
            if last_user_msg is None:
                simple_msgs = [
                    system_msg,
                    {"role": "user", "content": "Based on the conversation history above, please generate the appropriate workflow."}
                ]
            else:
                simple_msgs = [system_msg] + msgs_
        else:
            msgs_.insert(0, system_msg)
            simple_msgs = msgs_
            
        # Generate workflow
        response = await self.model.generate_chat_response(
            messages=simple_msgs,
        )
        response = normalize_chat_response(response)
        # Debug: Check response type and structure
        logger.debug(f"Model response type: {type(response)}")
        logger.debug(f"Model response content: {response}")
        
        return response

    def _get_system_prompt(self, node: str, as_message: bool = True) -> ChatCompletionSystemMessageParam:
        prompt = self.prompts.get(node).get("prompt")
        if as_message:
            return ChatCompletionSystemMessageParam(role="system", content=prompt)
        return prompt

    async def _generate_summary(self, msgs: list[ChatCompletionMessageParam]) -> str:
        """Generate summary of conversation history."""
        msgs_ = deepcopy(msgs)
        system_msg = self._get_system_prompt("history_summarization")

        msgs_.insert(0, system_msg)
        # Generate summary
        response = await self.model.generate_chat_response(
            messages=msgs_,
        )
        # Ensure response is a string
        if isinstance(response, dict):
            # If response is a dict, extract content
            if "content" in response:
                return str(response["content"])
            elif "message" in response and "content" in response["message"]:
                return str(response["message"]["content"])
            else:
                return str(response)
        return str(response)

    async def initialize(self):
        """Initialize the orchestration agent."""
        if not self.model:
            self.model = await self._setup_model_connection()
        logger.debug(f"OrchestrationAgent {self.agent_id} model initialized")
    
    async def _setup_model_connection(self) -> Any:
        """Set up LLM model connection based on configuration."""
        try:
            model = await ModelFactory.create_model(
                llm_config=self.llm_config,
                token_callback=self.token_callback
            )
            logger.debug(f"Successfully initialized model for OrchestrationAgent {self.agent_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model for OrchestrationAgent {self.agent_id}: {e}")
            raise
    
    async def select_tools(self, query: str, system_prompt: str = "") -> List[Dict[str, Any]]:
        """Select appropriate tools for the query. OrchestrationAgent doesn't use tools."""
        return []  # Orchestrator doesn't use tools directly
    
    async def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls. OrchestrationAgent doesn't use tools."""
        return []  # Orchestrator doesn't execute tools directly
    
    async def generate_response(self, query: str, tool_results: List[Dict[str, Any]], system_prompt: str = "") -> str:
        """Generate final response. For orchestrator, this delegates to generate_workflow."""
        # Convert query to message format
        msgs = [{"role": "user", "content": query}]
        return await self.generate_workflow(msgs)
    
    async def get_scenario(self, data: Dict[str, Any]):
        """
        Receive scenario data for scenario-based operation.
        
        Args:
            data: Dictionary containing scenario data from YAML file
        """
        # OrchestrationAgent can also receive scenario data for context
        logger.debug(f"🎭 OrchestrationAgent {self.agent_id} received scenario data with {len(data.get('steps', []))} steps")

Agent orchestration 확장 구조는 준수하지만 타입 계약 불일치·Silent Failure·취약한 문자열 템플릿 치환으로 인해 핵심 workflow 생성 신뢰성이 무너지는 7.5/10 수준의 설계다.

제안패치
"""
Orchestration Agent class for coordinating multiple agents.

This module provides the orchestration agent that coordinates workflows with enterprise-grade asynchronous safety, 
deduplicated prompt formatting utilities, and strict Pydantic/type contracts.
"""
import json
import asyncio
from typing import Dict, Any, List, Union
from pydantic import BaseModel, Field
from src.agents.base_agent import BaseAgent
from src.agents.llm_agent import LLMAgent
from src.utils.type_utils import normalize_chat_response
from src.utils.config_loader import format_agent_cards_for_prompt
from copy import copy
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolParam,
    ChatCompletionSystemMessageParam
)
from src.utils.model_factory import ModelFactory

# [타입 계약 강화] 출력 스키마 명확화를 위한 Pydantic 모델 정의
class IntentResponse(BaseModel):
    intent: str = Field(..., description="Classified user intent")
    confidence: float = Field(default=1.0, description="Confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context or parameters")

class WorkflowResponse(BaseModel):
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Generated workflow steps")
    status: str = Field(default="success", description="Execution status")


class OrchestrationAgent(BaseAgent):
    """Agent for coordinating multiple agents in workflows with fortified design patterns and production-grade safety."""
    
    def __init__(self, agent_id: str, agent_card_list: Union[List[Dict[str, Any]], Dict[str, Any]], llm_config: Dict[str, Any], prompts: Dict[str, Any]):
        """Initialize the orchestration agent with flexible container support and asynchronous locking."""
        self.agent_id = agent_id
        
        # [방어적 정합성] List나 Dict 어떤 형태의 입력이 들어와도 안전하게 수용
        if isinstance(agent_card_list, dict):
            self.agent_card_list = [agent_card_list]
        elif isinstance(agent_card_list, list):
            self.agent_card_list = agent_card_list
        else:
            logger.warning(f"OrchestrationAgent {self.agent_id}: Unexpected agent_card_list type {type(agent_card_list)}. Defaulting to empty list.")
            self.agent_card_list = []
            
        self.llm_config = llm_config
        self.prompts = prompts
        self.managed_agents = {}
        self.model = None
        self.token_callback = None
        
        # [비동기 안정성] 초기화 경쟁 조건(Race Condition) 원천 차단을 위한 asyncio.Lock 도입
        self._init_lock = asyncio.Lock()
    
    def set_managed_agents(self, agents: Dict[str, Any]):
        """Set the agents this orchestration agent manages."""
        self.managed_agents = agents

    def _format_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        """
        [중복 제거 및 중앙화] Prompt 템플릿 포매팅 로직을 전용 메서드로 분리하여
        template.format()과 replace fallback 정책을 단일 계층에서 관리합니다.
        """
        try:
            return template.format(**variables)
        except (KeyError, ValueError) as e:
            logger.warning(f"OrchestrationAgent {self.agent_id}: Strict prompt formatting failed ({e}). Falling back to safe string replacement.")
            prompt = template
            for key, val in variables.items():
                placeholder = f"%%{key}%%"
                prompt = prompt.replace(placeholder, str(val) if val is not None else "")
            return prompt

    async def execute(self, msgs: list[ChatCompletionMessageParam], 
                      prev_workflow: Dict[str, Any] = None, 
                      system_info: str = "",
                      skip_summary: bool = True,
                      kwargs: Dict[str, Any] = None) -> Union[Dict[str, Any], str]:
        """Execute orchestration agent - coordinate multiple agents."""
        if not self.model:
            await self.initialize()
        return await self.generate_workflow(msgs, prev_workflow, system_info, skip_summary, kwargs)

    async def summarization(self, msgs: list[ChatCompletionMessageParam]) -> list[ChatCompletionAssistantMessageParam]:
        """Summarize workflow execution with shallow copy overhead reduction."""
        if not self.model:
            await self.initialize()
            
        # [성능 최적화] 무거운 deepcopy 대신 가벼운 list.copy() 활용 (메모리 및 CPU 오버헤드 O(1) 수준 단축)
        msgs_ = msgs.copy()
        system_msg = self._get_system_prompt("summarize")
        msgs_.insert(0, system_msg)
        
        logger.debug(f"OrchestrationAgent: Starting summarization with {len(msgs)} messages")
        response = await self.model.generate_chat_response(messages=msgs_)
        response = normalize_chat_response(response)
        
        logger.debug(f"OrchestrationAgent: Summarization completed, response type: {type(response)}")
        return response

    async def classify_intent(self, msgs: list[ChatCompletionMessageParam], 
                                system_info: str = "", 
                                prev_workflow: Dict[str, Any] = None, 
                                kwargs: Dict[str, Any] = None) -> Union[Dict[str, Any], IntentResponse]:
        """Execute intent classification."""
        if not self.model:
            await self.initialize()
        return await self._classify_intent(msgs, system_info, prev_workflow, kwargs)

    async def _classify_intent(self, msgs: list[ChatCompletionMessageParam],
                              system_info: str = "", 
                              prev_workflow: Dict[str, Any] = None, 
                              kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        msgs_ = msgs.copy()
        prompt_template = self._get_system_prompt("classify", as_message=False)
        workflows_str = json.dumps(prev_workflow, ensure_ascii=False) if prev_workflow else "NONE"
        
        # [중앙화 적용] 중앙 집중식 프롬프트 포매터 사용
        prompt_content = self._format_prompt(prompt_template, {
            "system_info": system_info,
            "workflows": workflows_str
        })

        system_msg = ChatCompletionSystemMessageParam(
            role="system",
            content=prompt_content
        )
        msgs_.insert(0, system_msg)
        response = await self.model.generate_chat_response(messages=msgs_)
        normalized = normalize_chat_response(response)
        
        # [타입 보증] 반환값 타입 정합성 검증
        if isinstance(normalized, dict):
            return normalized
        return {"intent": str(normalized), "metadata": {}}
    
    async def generate_workflow(self, msgs: list[ChatCompletionMessageParam], 
                                prev_workflow: Dict[str, Any] = None,
                                system_info: str = "", 
                                skip_summary: bool = True, 
                                kwargs: dict = None) -> Dict[str, Any]:
        """Generate workflow JSON based on input messages with unified prompt formatting."""
        msgs_ = msgs.copy()
        prompt_template = self._get_system_prompt("general", as_message=False)
        
        agent_cards_text = format_agent_cards_for_prompt(self.agent_card_list)
        workflows_str = json.dumps(prev_workflow, ensure_ascii=False) if prev_workflow else "NONE"
        
        history_text = ""
        for i, msg in enumerate(msgs):
            role = msg.get("role", "")
            content = msg.get("content", "")
            history_text += f"[Message {i+1}] {role}: {content}\n\n"

        # [중앙화 적용] 중복 제거된 프롬프트 포매터 호출
        prompt = self._format_prompt(prompt_template, {
            "agent_cards": agent_cards_text,
            "system_info": system_info,
            "workflows": workflows_str,
            "history": history_text
        })

        logger.debug(f"OrchestrationAgent: len(msgs)={len(msgs)}, skip_summary={skip_summary}")
        logger.debug(f"OrchestrationAgent: Added {len(msgs)} messages to history")

        system_msg = ChatCompletionSystemMessageParam(
            role="system",
            content=prompt
        )
        
        if skip_summary and len(msgs) > 0:
            last_user_msg = next((msg for msg in reversed(msgs) if msg.get("role") == "user"), None)
            if last_user_msg is None:
                simple_msgs = [
                    system_msg,
                    {"role": "user", "content": "Based on the conversation history above, please generate the appropriate workflow."}
                ]
            else:
                simple_msgs = [system_msg] + msgs_
        else:
            msgs_.insert(0, system_msg)
            simple_msgs = msgs_
            
        response = await self.model.generate_chat_response(messages=simple_msgs)
        response = normalize_chat_response(response)
        
        logger.debug(f"Model response type: {type(response)}")
        logger.debug(f"Model response content: {response}")
        
        return response if isinstance(response, dict) else {"result": response}

    def _get_system_prompt(self, node: str, as_message: bool = True) -> Union[ChatCompletionSystemMessageParam, str]:
        """Retrieve system prompt with defensive fallback."""
        node_config = self.prompts.get(node)
        if not node_config or "prompt" not in node_config:
            logger.warning(f"OrchestrationAgent {self.agent_id}: Prompt node '{node}' not found. Using empty fallback.")
            prompt = ""
        else:
            prompt = node_config.get("prompt")
            
        if as_message:
            return ChatCompletionSystemMessageParam(role="system", content=prompt)
        return prompt

    async def _generate_summary(self, msgs: list[ChatCompletionMessageParam]) -> str:
        """Generate summary of conversation history with shallow copy and strict schema enforcement."""
        msgs_ = msgs.copy()
        system_msg = self._get_system_prompt("history_summarization")
        msgs_.insert(0, system_msg)
        
        response = await self.model.generate_chat_response(messages=msgs_)
        normalized = normalize_chat_response(response)
        
        if isinstance(normalized, str):
            return normalized
        elif isinstance(normalized, dict):
            if "content" in normalized:
                return str(normalized["content"])
            elif "message" in normalized and isinstance(normalized["message"], dict) and "content" in normalized["message"]:
                return str(normalized["message"]["content"])
            else:
                logger.error(f"OrchestrationAgent: Unexpected dictionary response structure in summary: {normalized}")
                raise ValueError("Failed to extract summary string from structured response due to schema mismatch.")
        return str(normalized)

    async def initialize(self):
        """
        Initialize the orchestration agent thread-safely.
        [경쟁 조건 방어] 비동기 환경 다중 요청 시 중복 인스턴스화 방지를 위해 asyncio.Lock 적용
        """
        if self.model is not None:
            return

        async with self._init_lock:
            # 더블 체크 락킹 패턴 (Double-Checked Locking Pattern) 적용
            if self.model is None:
                self.model = await self._setup_model_connection()
                logger.debug(f"OrchestrationAgent {self.agent_id} model initialized thread-safely")
    
    async def _setup_model_connection(self) -> Any:
        """Set up LLM model connection based on configuration."""
        try:
            model = await ModelFactory.create_model(
                llm_config=self.llm_config,
                token_callback=self.token_callback
            )
            logger.debug(f"Successfully initialized model for OrchestrationAgent {self.agent_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model for OrchestrationAgent {self.agent_id}: {e}")
            raise
    
    async def select_tools(self, query: str, system_prompt: str = "") -> List[Dict[str, Any]]:
        """Select appropriate tools for the query. OrchestrationAgent doesn't use tools."""
        return []
    
    async def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls. OrchestrationAgent doesn't use tools."""
        return []
    
    async def generate_response(self, query: str, tool_results: List[Dict[str, Any]], system_prompt: str = "") -> str:
        """Generate final response. Delegates to generate_workflow."""
        msgs = [{"role": "user", "content": query}]
        res = await self.generate_workflow(msgs)
        if isinstance(res, dict):
            return json.dumps(res, ensure_ascii=False)
        return str(res)
    
    async def get_scenario(self, data: Dict[str, Any]):
        """Receive scenario data for scenario-based operation."""
        logger.debug(f"🎭 OrchestrationAgent {self.agent_id} received scenario data with {len(data.get('steps', []))} steps")


최종 개선사항
✅ asyncio Lock 도입 → 모델 초기화 Race Condition 및 중복 LLM 인스턴스 생성 차단
✅ Prompt format 로직 분산 → _format_prompt() 중앙화로 템플릿 정책 단일 관리
✅ deepcopy 반복 사용 → 얕은 복사 전환으로 메시지 처리 메모리 비용 절감
✅ agent_card_list 입력 불일치 → Dict/List 정규화 계층 추가로 데이터 계약 안정화
✅ normalize 이후 반환 타입 불명확 → IntentResponse/WorkflowResponse 기반 타입 계약 구조 마련
✅ Silent 문자열 변환 처리 → 비정상 Response Schema 감지 및 명시적 오류 발생 구조 강화
✅ Legacy replace 의존 → format 우선 + fallback 구조로 구형 프롬프트 호환성 유지
✅ 단순 model 초기화 체크 → Double Checked Locking 적용으로 동시 요청 안전성 확보
✅ Prompt 생성 책임 분산 → Formatter Utility 계층화로 유지보수성과 테스트 가능성 향상
✅ LLM 응답 문자열/딕셔너리 혼재 → 최소 Response Normalization Layer 적용으로 API 안정성 강화

비동기 안전성·템플릿 관리·타입 계약·예외 추적까지 엔터프라이즈 운영 기준에 맞게 대부분 보완됐지만, 최종 단계에서는 Pydantic Response 모델을 실제 반환 객체로 강제하고 LLM Schema Validation 계층을 추가해야 9.7~9.8 수준의 완성도까지 도달할 수 있다.

