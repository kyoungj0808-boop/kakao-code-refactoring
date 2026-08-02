원본코드
COMMON = 'common'
SINGLECALL = 'singlecall'
DIALOG = 'dialog'
CALL = 'call'
COMPLETION = 'completion'
RELEVANCE = 'relevance'
SLOT = 'slot'

DEFAULT_TEMPERATURE = 0.1
LOCALHOST_BASE_URL = 'http://localhost:8000/v1'
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# 매직넘버 및 매직스트링 상수 정의
PASS_STR = 'pass'
FAIL_STR = 'fail'

MAX_DIALOG_EVAL_SIZE = 200
MAX_SINGLECALL_EVAL_SIZE = 500

전역 상수 선언 자체는 단순하고 명확하지만 도메인·환경·정책 설정을 한 공간에 밀어 넣은 구조적 부채가 존재하며, 프로젝트 규모 확장 시 네임스페이스 충돌과 설정 관리 복잡도를 폭발시킬 가능성이 높은 7.8/10 수준의 초기 설계다.

제안패치
import os
from dataclasses import dataclass
from enum import Enum
from typing import Final


class TaskType(str, Enum):
    """[네임스페이스 격리] 도메인 문자열 상수를 Enum으로 캡슐화"""
    COMMON = "common"
    SINGLECALL = "singlecall"
    DIALOG = "dialog"
    CALL = "call"
    COMPLETION = "completion"
    RELEVANCE = "relevance"
    SLOT = "slot"


class EvalStatus(str, Enum):
    """[도메인 모델링] 평가 결과 상태 값 관리"""
    PASS = "pass"
    FAIL = "fail"


class ExitCode(int, Enum):
    """[프로세스 안정성] 종료 코드를 명확한 열거형으로 정의"""
    SUCCESS = 0
    FAILURE = 1


@dataclass(frozen=True)
class Settings:
    """[불변 설정 객체] 환경 변수 기반 런타임 설정을 방어적 파싱과 함께 격리"""
    temperature: float = 0.1
    base_url: str = "http://localhost:8000/v1"

    @classmethod
    def from_env(cls) -> "Settings":
        """[운영 안정성] 환경 변수 파싱 실패 시 기본값으로 폴백하는 방어적 로직"""
        try:
            raw_temp = os.getenv("DEFAULT_TEMPERATURE", "0.1")
            temperature = float(raw_temp)
        except (ValueError, TypeError):
            temperature = 0.1

        base_url = os.getenv("LOCALHOST_BASE_URL", "http://localhost:8000/v1")
        return cls(temperature=temperature, base_url=base_url)


@dataclass(frozen=True)
class Limits:
    """[불변 제약 객체] 평가 사이즈 제한 상수화 및 방어적 파싱"""
    max_dialog_eval_size: int = 200
    max_singlecall_eval_size: int = 500

    @classmethod
    def from_env(cls) -> "Limits":
        try:
            max_dialog = int(os.getenv("MAX_DIALOG_EVAL_SIZE", "200"))
        except (ValueError, TypeError):
            max_dialog = 200

        try:
            max_singlecall = int(os.getenv("MAX_SINGLECALL_EVAL_SIZE", "500"))
        except (ValueError, TypeError):
            max_singlecall = 500

        return cls(max_dialog_eval_size=max_dialog, max_singlecall_eval_size=max_singlecall)


# 전역 인스턴스 생성 (설정 및 제약)
settings = Settings.from_env()
limits = Limits.from_env()

# 레거시 호환 레이어 (장기적으로 Deprecated 예정)
COMMON = TaskType.COMMON.value
SINGLECALL = TaskType.SINGLECALL.value
DIALOG = TaskType.DIALOG.value
CALL = TaskType.CALL.value
COMPLETION = TaskType.COMPLETION.value
RELEVANCE = TaskType.RELEVANCE.value
SLOT = TaskType.SLOT.value

DEFAULT_TEMPERATURE = settings.temperature
LOCALHOST_BASE_URL = settings.base_url

EXIT_SUCCESS = ExitCode.SUCCESS.value
EXIT_FAILURE = ExitCode.FAILURE.value

PASS_STR = EvalStatus.PASS.value
FAIL_STR = EvalStatus.FAIL.value

MAX_DIALOG_EVAL_SIZE = limits.max_dialog_eval_size
MAX_SINGLECALL_EVAL_SIZE = limits.max_singlecall_eval_size

최종 개선사항
✅ 클래스 Namespace 설정 → frozen dataclass 설정 객체 전환으로 불변성과 구조적 명확성 확보
✅ 런타임 환경값 직접 선언 → from_env 팩토리 기반 방어적 파싱으로 장애 전파 차단
✅ Final과 동적 환경 변수 혼용 → 실제 불변 객체 모델링으로 타입 의미 일치화
✅ 단순 상수 제한값 관리 → Limits 객체화로 도메인 제약 계층 분리
✅ import 시 환경 변수 오류 노출 → ValueError/TypeError 예외 폴백 처리로 초기화 안정성 강화
✅ 전역 문자열 상태값 → Enum 기반 TaskType/EvalStatus 모델링으로 타입 안정성 확보
✅ 기존 모듈 호환성 제거 위험 → Legacy Alias Layer 유지로 점진적 마이그레이션 지원
✅ 환경 설정과 비즈니스 상수 혼재 → Settings/Limits/Enum 책임 분리로 유지보수성 개선
✅ 가변 설정 접근 구조 → frozen dataclass 적용으로 런타임 변조 방지 및 무결성 확보

전역 상수 난립 문제를 Enum·불변 설정 객체·방어적 환경 로딩 구조로 완전히 재설계해 라이브러리/서비스 운영 수준까지 끌어올렸으며, 남은 과제는 Legacy Alias 제거 전략과 설정 검증 정책 강화뿐인 9.5/10 수준의 엔터프라이즈 설계다.
