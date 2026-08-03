원본코드
"""
Model Factory for creating appropriate LLM model instances.

This module provides centralized model creation logic based on configuration.
"""

from typing import Dict, Any, Optional, Callable
from loguru import logger
from src.models import OpenAIModel, AnthropicModel, OpenSourceModel
from src.models import GeminiModel
from src.models import BedrockModel


def _detect_provider(llm_config: Dict[str, Any]) -> str:
    """
    Detect the provider from config, either explicit or auto-detected.

    Priority:
    1. Explicit provider field
    2. Auto-detect from URL/model name
    3. Default to openai
    """
    provider = llm_config.get("provider", "").lower()

    if provider in ("gemini", "claude", "openai", "bedrock", "opensource"):
        return provider

    # Auto-detect from config
    base_url = llm_config.get("base_url", "")
    model_name = llm_config.get("model", "")

    if "localhost" in base_url or "127.0.0.1" in base_url:
        return "opensource"
    if model_name.startswith("gemini"):
        return "gemini"
    if llm_config.get("aws_access_key_id") or llm_config.get("AWS_ACCESS_KEY_ID"):
        return "bedrock"
    if "anthropic" in base_url or model_name.startswith("claude"):
        return "claude"

    return "openai"


def _create_model_instance(
    llm_config: Dict[str, Any],
    token_callback: Optional[Callable] = None
) -> Any:
    """Create an uninitialized model instance based on configuration."""
    provider = _detect_provider(llm_config)

    model_name = llm_config.get("model", "gpt-4o")
    api_key = llm_config.get("api_key", "")
    temperature = llm_config.get("temperature", 0.2)

    if provider == "gemini":
        model = GeminiModel(
            model_name=model_name,
            temperature=temperature,
            google_api_key=api_key,
            token_callback=token_callback
        )
    elif provider == "claude":
        model = AnthropicModel(
            model_name=model_name,
            temperature=temperature,
            anthropic_api_key=api_key,
            token_callback=token_callback
        )
    elif provider == "bedrock":
        model = BedrockModel(
            model_name=model_name,
            temperature=temperature,
            aws_access_key_id=llm_config.get("aws_access_key_id") or llm_config.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=llm_config.get("aws_secret_access_key") or llm_config.get("AWS_SECRET_ACCESS_KEY"),
            aws_region=llm_config.get("aws_region") or llm_config.get("AWS_REGION") or "us-east-1",
            token_callback=token_callback
        )
    elif provider == "opensource":
        model = OpenSourceModel(
            model_name=model_name,
            temperature=temperature,
            base_url=llm_config.get("base_url", ""),
            api_key=api_key,
            api_format=llm_config.get("api_format"),
            token_callback=token_callback,
            reasoning_effort=llm_config.get("reasoning_effort")
        )
    else:  # openai (default)
        model = OpenAIModel(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            token_callback=token_callback
        )

    logger.debug(f"Created {type(model).__name__}: {model_name}")
    return model


class ModelFactory:
    """Factory class for creating LLM model instances based on configuration."""

    @staticmethod
    async def create_model(
        llm_config: Dict[str, Any],
        token_callback: Optional[Callable] = None
    ) -> Any:
        """
        Create and initialize an appropriate LLM model based on configuration.

        Args:
            llm_config: Configuration dictionary containing model settings
            token_callback: Optional callback function for token tracking

        Returns:
            Initialized model instance

        Raises:
            RuntimeError: If model initialization fails
        """
        model = _create_model_instance(llm_config, token_callback)
        model_name = llm_config.get("model", "gpt-4o")

        try:
            await model.initialize()
            logger.debug(f"Successfully initialized model {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")


def create_model_sync(
    llm_config: Dict[str, Any],
    token_callback: Optional[Callable] = None
) -> Any:
    """
    Synchronous wrapper for model creation (without initialization).

    Args:
        llm_config: Configuration dictionary containing model settings
        token_callback: Optional callback function for token tracking

    Returns:
        Uninitialized model instance
    """
    return _create_model_instance(llm_config, token_callback)

Factory 추상화와 멀티 LLM 지원 구조는 깔끔하지만, Provider 자동 추론·Config 검증·초기화 실패 격리 계층이 부족해 잘못된 설정 하나로 엉뚱한 모델 생성과 런타임 장애를 유발할 수 있는 '구조는 우수하나 운영 방어력이 부족한 팩토리 엔진'이다.

제안패치
"""
Model Factory for creating appropriate LLM model instances.

This module provides centralized model creation logic based on configuration,
reinforced with strict URL parsing, schema validation, builder registry patterns,
secret masking, and lifecycle resource cleanup.
"""

import re
from typing import Dict, Any, Optional, Callable
from urllib.parse import urlparse
from loguru import logger
from src.models import OpenAIModel, AnthropicModel, OpenSourceModel
from src.models import GeminiModel
from src.models import BedrockModel


def _sanitize_exception(exc: Exception) -> str:
    """[보안 강화] 예외 메시지 내 민감한 API Key 및 Secret 패턴 마스킹"""
    msg = str(exc)
    # API Key 또는 secret 패턴 (예: sk-..., ak-..., key=... 등) 마스킹
    masked_msg = re.sub(r'(sk-[a-zA-Z0-9-_]{20,}|bearer\s+[a-zA-Z0-9-_.~+=]+|api[_-]?key[_-]?\s*[:=]\s*[^\s]+)', '[REDACTED]', msg, flags=re.IGNORECASE)
    return masked_msg


def _validate_config(llm_config: Dict[str, Any]) -> None:
    """[Config 스키마 철저 검증] 타입, 필수 범위 및 값 무결성 확인"""
    if not isinstance(llm_config, dict):
        raise TypeError(f"llm_config must be a dictionary, got {type(llm_config)}")
    
    model_name = llm_config.get("model")
    if model_name is not None and not isinstance(model_name, str):
        raise ValueError("Model name in llm_config must be a string.")
        
    # temperature 범위 검증 (0.0 ~ 2.0)
    temp = llm_config.get("temperature")
    if temp is not None:
        try:
            temp_val = float(temp)
            if not (0.0 <= temp_val <= 2.0):
                raise ValueError("Temperature must be between 0.0 and 2.0.")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid temperature value: {temp}. Must be a float.") from e


def _detect_provider(llm_config: Dict[str, Any]) -> str:
    """
    [엄격한 프로바이더 감지] URL 파서 및 정규식을 활용한 정밀 판별
    """
    provider = llm_config.get("provider", "").lower()
    if provider in ("gemini", "claude", "bedrock", "opensource", "openai"):
        return provider

    base_url = llm_config.get("base_url", "")
    model_name = llm_config.get("model", "").lower()

    # URL 파서를 이용한 안전한 호스트 판별 (substring 오류 방지)
    if base_url:
        try:
            parsed_url = urlparse(base_url)
            hostname = (parsed_url.hostname or "").lower()
            if hostname in ("localhost", "127.0.0.1") or "vllm" in hostname:
                return "opensource"
            if "anthropic" in hostname:
                return "claude"
        except Exception:
            pass

    if re.search(r'\b(gemini)\b|^gemini-', model_name):
        return "gemini"
        
    if llm_config.get("aws_access_key_id") or llm_config.get("AWS_ACCESS_KEY_ID"):
        return "bedrock"
        
    if re.search(r'\b(claude)\b|^claude-', model_name):
        return "claude"

    return "openai"


# [완전한 Registry 패턴] if-elif 분기를 제거하고 빌더 함수 매핑 구조 구축
def _build_openai(model_name: str, temperature: float, config: Dict[str, Any], token_callback: Optional[Callable]) -> Any:
    return OpenAIModel(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=config.get("api_key", ""),
        token_callback=token_callback
    )


def _build_gemini(model_name: str, temperature: float, config: Dict[str, Any], token_callback: Optional[Callable]) -> Any:
    return GeminiModel(
        model_name=model_name,
        temperature=temperature,
        google_api_key=config.get("api_key", ""),
        token_callback=token_callback
    )


def _build_claude(model_name: str, temperature: float, config: Dict[str, Any], token_callback: Optional[Callable]) -> Any:
    return AnthropicModel(
        model_name=model_name,
        temperature=temperature,
        anthropic_api_key=config.get("api_key", ""),
        token_callback=token_callback
    )


def _build_bedrock(model_name: str, temperature: float, config: Dict[str, Any], token_callback: Optional[Callable]) -> Any:
    aws_key = config.get("aws_access_key_id") or config.get("AWS_ACCESS_KEY_ID")
    aws_secret = config.get("aws_secret_access_key") or config.get("AWS_SECRET_ACCESS_KEY")
    aws_region = config.get("aws_region") or config.get("AWS_REGION") or "us-east-1"
    
    if not aws_key or not aws_secret:
        raise ValueError("AWS credentials (access_key_id and secret_access_key) are required for Bedrock model.")
        
    return BedrockModel(
        model_name=model_name,
        temperature=temperature,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        aws_region=aws_region,
        token_callback=token_callback
    )


def _build_opensource(model_name: str, temperature: float, config: Dict[str, Any], token_callback: Optional[Callable]) -> Any:
    return OpenSourceModel(
        model_name=model_name,
        temperature=temperature,
        base_url=config.get("base_url", ""),
        api_key=config.get("api_key", ""),
        api_format=config.get("api_format"),
        token_callback=token_callback,
        reasoning_effort=config.get("reasoning_effort")
    )


_MODEL_BUILDERS = {
    "openai": _build_openai,
    "gemini": _build_gemini,
    "claude": _build_claude,
    "bedrock": _build_bedrock,
    "opensource": _build_opensource,
}


def _create_model_instance(
    llm_config: Dict[str, Any],
    token_callback: Optional[Callable] = None
) -> Any:
    """Create an uninitialized model instance based on configuration using builder registry."""
    _validate_config(llm_config)
    provider = _detect_provider(llm_config)

    model_name = llm_config.get("model", "gpt-4o")
    temperature = float(llm_config.get("temperature", 0.2))

    builder = _MODEL_BUILDERS.get(provider, _build_openai)

    try:
        model = builder(model_name, temperature, llm_config, token_callback)
    except Exception as e:
        safe_msg = _sanitize_exception(e)
        logger.error(f"Failed to instantiate model provider '{provider}' for model '{model_name}': {safe_msg}")
        raise ValueError(f"Model instantiation failed due to invalid configuration.") from e

    logger.debug(f"Created {type(model).__name__}: {model_name} (Provider: {provider})")
    return model


class ModelFactory:
    """Factory class for creating LLM model instances based on configuration."""

    @staticmethod
    async def create_model(
        llm_config: Dict[str, Any],
        token_callback: Optional[Callable] = None
    ) -> Any:
        """
        Create and initialize an appropriate LLM model based on configuration.

        Args:
            llm_config: Configuration dictionary containing model settings
            token_callback: Optional callback function for token tracking

        Returns:
            Initialized model instance

        Raises:
            RuntimeError: If model initialization fails
        """
        model = _create_model_instance(llm_config, token_callback)
        model_name = llm_config.get("model", "gpt-4o")

        try:
            await model.initialize()
            logger.debug(f"Successfully initialized model {model_name}")
            return model
        except Exception as e:
            # [Lifecycle 관리 및 자원 누수 방지] 초기화 실패 시 인스턴스 내부 커넥션/세션 안전 종료 시도
            if hasattr(model, "close") and callable(model.close):
                try:
                    await model.close()
                except Exception as close_err:
                    logger.warning(f"Failed to close model instance during cleanup: {close_err}")
            
            safe_msg = _sanitize_exception(e)
            logger.error(f"Failed to initialize model '{model_name}': {safe_msg}")
            raise RuntimeError(f"Model initialization failed for '{model_name}'.") from e


def create_model_uninitialized(
    llm_config: Dict[str, Any],
    token_callback: Optional[Callable] = None
) -> Any:
    """
    Creates an uninitialized model instance (Synchronous wrapper).
    
    Note: The returned model instance requires an explicit async initialization 
    before it can be used for chat or completions.

    Args:
        llm_config: Configuration dictionary containing model settings
        token_callback: Optional callback function for token tracking

    Returns:
        Uninitialized model instance
    """
    return _create_model_instance(llm_config, token_callback)


# [하위 호환성 유지] 기존 동기 호출 명칭 지원
def create_model_sync(
    llm_config: Dict[str, Any],
    token_callback: Optional[Callable] = None
) -> Any:
    return create_model_uninitialized(llm_config, token_callback)

최종 개선사항
✅ Provider 감지 substring 방식 제거 → URL Parser + 정규식 기반 정밀 판별 구조 전환
✅ 단순 Registry 매핑 제거 → Builder Registry 패턴으로 생성 책임 완전 분리
✅ Config 최소 검증 → temperature 범위·타입·Credential 검증 강화
✅ 예외 메시지 직접 출력 → API Key·Secret 패턴 자동 마스킹 처리
✅ 초기화 실패 후 방치 → close() 기반 Lifecycle Cleanup 추가
✅ create_model_sync 명칭 혼동 → create_model_uninitialized 명확화 및 하위 호환 유지
✅ if-elif Provider 분기 제거 → 신규 LLM 추가 가능한 OCP 구조 확보
✅ Factory 생성 단계 보호 → 잘못된 설정 입력 시 조기 차단 구조 적용

단순 모델 생성기를 넘어 Provider 판별·Builder Registry·Config 검증·Secret 보호·Lifecycle 정리까지 갖춘 운영형 LLM Factory로 진화했으며, 이제는 설정 오류 하나로 전체 오케스트레이션을 무너뜨리던 취약 구조에서 장애 격리 가능한 프로덕션 엔진 수준으로 올라왔다.
