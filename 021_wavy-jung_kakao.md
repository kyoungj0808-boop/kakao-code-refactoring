원본코드
import re
from typing import Any, Dict, List, Tuple

from .__base__ import MinimalTemplate


class XSumMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "xsum_minimal"

    def get_answer_choices_list(self, example: Any) -> List[str]:
        return example["summary"]

    def apply(self, example: Any) -> Tuple[str]:
        demon = f"Article: {example['document']}\nShort summary:"
        label = example["summary"]
        return (demon, label)

MinimalTemplate 기반 설계 방향은 적절하지만, 반환 계약 불일치·입력 검증 부재·모호한 네이밍으로 인해 안정성을 희생한 연구용 프로토타입 수준의 템플릿이다.

제안패치
from typing import Any, List, Tuple

from .__base__ import MinimalTemplate


class XSumMinimal(MinimalTemplate):
    def __init__(self):
        self.name = "xsum_minimal"

    def get_answer_choices_list(self, example: Any) -> List[str]:
        """
        [데이터 무결성 강화] 자동 형변환을 제거하고 엄격한 타입 검증을 거쳐 
        오염된 데이터가 후보군으로 주입되는 조용한 실패(Silent Failure) 원천 차단
        """
        if not isinstance(example, dict):
            raise TypeError(f"Expected dict for example, got {type(example)}")
        
        summary = example.get("summary")
        if summary is None:
            raise KeyError("Required key 'summary' is missing or None in example.")
        
        # [엄격한 타입 정합성] 문자열 타입만 허용하여 잘못된 객체 형변환 방지
        if not isinstance(summary, str):
            raise TypeError(f"Expected 'summary' to be str, got {type(summary)}")
        
        return [summary]

    def apply(self, example: Any) -> Tuple[str, str]:
        """
        [무결성 및 안정성 최종 완성] 불필요한 import 제거 및 엄격한 타입 검증 적용
        """
        if not isinstance(example, dict):
            raise TypeError(f"Expected dict for example, got {type(example)}")

        document = example.get("document")
        summary = example.get("summary")

        if document is None or summary is None:
            raise KeyError(f"Missing required keys ('document' or 'summary') in example: {list(example.keys())}")

        if not isinstance(document, str):
            raise TypeError(f"Expected 'document' to be str, got {type(document)}")
        if not isinstance(summary, str):
            raise TypeError(f"Expected 'summary' to be str, got {type(summary)}")

        prompt = f"Article: {document}\nShort summary:"
        label = summary
        
        return (prompt, label)

최종 개선사항
✅ 자동 str() 형변환 제거 → 잘못된 데이터가 학습 데이터로 유입되는 Silent Failure 차단
✅ summary 다중 타입 허용 → XSum 데이터 스키마 기준 str 단일 타입 검증 구조로 강화
✅ document 입력 검증 추가 → 비정상 데이터 타입의 Prompt 오염 방지
✅ 반환 타입 선언 → 실제 (prompt, label) 구조와 Tuple[str, str] 계약 일치
✅ 불필요 import 제거 → 정적 분석 경고 제거 및 코드 의존성 최소화
✅ 변수명 demon 제거 → prompt 변경으로 의도 명확화 및 유지보수성 향상
✅ answer choice 생성 로직 → 단일 정답 요약 구조에 맞춘 엄격한 후보 생성 방식으로 변경
✅ 입력 예외 처리 → 암묵적 실패 대신 명시적 TypeError/KeyError 발생 구조 적용
✅ 데이터 파이프라인 안정성 → 오염 데이터 전파 차단 중심의 방어적 템플릿 구조 확보

연구용 최소 템플릿에서 벗어나 입력 스키마 검증·타입 계약·데이터 무결성을 갖춘 운영 가능한 NLP 데이터 어댑터 수준으로 완성되었으며, 남은 개선점은 공통 Schema Validator 추상화 정도다.
