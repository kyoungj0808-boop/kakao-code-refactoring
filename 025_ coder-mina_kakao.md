원본코드
import random

from n2 import HnswIndex

f = 3
t = HnswIndex(f, "L2")
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_data(v)

t.build(m=5, max_m0=10, n_threads=4)
t.save('test.n2')

u = HnswIndex(f, "L2")
u.load('test.n2')

search_id = 1
k = 3
neighbor_ids = u.search_by_id(search_id, k)
print(
    "[search_by_id]: Nearest neighborhoods of id {}: {}".format(
        search_id,
        neighbor_ids))

example_vector_query = [random.gauss(0, 1) for z in range(f)]
nns = u.search_by_vector(example_vector_query, k, include_distances=True)
print(
    "[search_by_vector]: Nearest neighborhoods of vector {}: {}".format(
        example_vector_query,
        nns))

공식 데모로서는 충분하지만, 프로덕션 기준에서는 재현성·자원 관리·원자적 저장을 보강해야 안정적으로 운영할 수 있다.

제안패치
import logging
import os
import random
from typing import List, Tuple
from n2 import HnswIndex

# [아키텍처 개선] 전역 상태 격리를 위한 독립 RNG 및 HNSW 하이퍼파라미터 상수화
DIMENSION: int = 3
NUM_ELEMENTS: int = 1000
TOP_K: int = 3
SEARCH_ID: int = 1
INDEX_PATH: str = "test.n2"
RANDOM_SEED: int = 42

# HNSW 세부 빌드 파라미터 상수 분리
HNSW_M: int = 5
HNSW_MAX_M0: int = 10
HNSW_THREADS: int = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("N2ProductionPipeline")


def build_and_save_index(index_path: str) -> None:
    """[독립 RNG 및 예외 구체화] 전역 상태 오염 방지 및 명시적 예외 처리가 적용된 인덱스 빌드 함수"""
    # [재현성 강화] 전역 random 모듈 대신 독립된 Random 객체 주입
    rng = random.Random(RANDOM_SEED)
    
    t = HnswIndex(DIMENSION, "L2")
    tmp_path = f"{index_path}.tmp"
    
    try:
        logger.info(f"Generating {NUM_ELEMENTS} vectors (dim={DIMENSION}) using isolated RNG...")
        for _ in range(NUM_ELEMENTS):
            v = [rng.gauss(0, 1) for _ in range(DIMENSION)]
            t.add_data(v)

        logger.info("Building HNSW index with configured parameters...")
        t.build(m=HNSW_M, max_m0=HNSW_MAX_M0, n_threads=HNSW_THREADS)
        
        # 원자적 저장을 위한 임시 파일 저장 후 rename
        t.save(tmp_path)
        os.replace(tmp_path, index_path)
        logger.info(f"Index successfully saved and committed to {index_path}")
        
    except (IOError, OSError) as e:
        logger.error(f"File I/O error during index save/replace: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    except RuntimeError as e:
        logger.error(f"N2 C++ execution error during index build: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    finally:
        # [자원 해제 보장] C++ 네이티브 메모리 누수 원천 차단
        try:
            t.unload()
        except Exception as unload_err:
            logger.warning(f"Failed to cleanly unload build index: {unload_err}")


def load_and_search_index(index_path: str) -> None:
    """[안전성 강화] 파일 존재 사전 검증, 독립 RNG 기반 쿼리 생성 및 확실한 자원 해제"""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    # [재현성 강화] 검색용 독립 RNG 객체 분리 (seed 오프셋 적용)
    query_rng = random.Random(RANDOM_SEED + 1)

    u = HnswIndex(DIMENSION, "L2")
    try:
        logger.info(f"Loading index from {index_path}...")
        u.load(index_path)

        # 1. ID 기반 검색
        neighbor_ids = u.search_by_id(SEARCH_ID, TOP_K)
        logger.info(f"[search_by_id]: Nearest neighborhoods of id {SEARCH_ID}: {neighbor_ids}")

        # 2. 벡터 기반 검색 (독립 RNG 사용)
        example_vector_query = [query_rng.gauss(0, 1) for _ in range(DIMENSION)]
        nns = u.search_by_vector(example_vector_query, TOP_K, include_distances=True)
        logger.info(f"[search_by_vector]: Nearest neighborhoods of vector {example_vector_query}: {nns}")

    except (IOError, OSError) as e:
        logger.error(f"File I/O error during index load: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"N2 C++ execution error during search: {e}")
        raise
    finally:
        # [자원 해제 보장]
        try:
            u.unload()
        except Exception as unload_err:
            logger.warning(f"Failed to cleanly unload search index: {unload_err}")


if __name__ == "__main__":
    build_and_save_index(INDEX_PATH)
    load_and_search_index(INDEX_PATH)

최종 개선사항
✅ 전역 random 제거 → 독립 random.Random() 객체 사용으로 재현성과 전역 상태 격리 확보
✅ HNSW 빌드 파라미터 하드코딩 제거 → HNSW_M, HNSW_MAX_M0, HNSW_THREADS 상수화
✅ 단순 save() → 임시 파일 저장 후 os.replace()를 통한 원자적 저장 적용
✅ finally에서 unload() 보장 → C++ 네이티브 리소스 누수 방지
✅ 광범위한 Exception 처리 → IOError, OSError, RuntimeError 중심의 구체적 예외 처리
✅ 검색용 RNG 분리 → 학습 데이터 생성과 검색 쿼리 생성의 독립성 확보
✅ 상세 로깅 추가 → 장애 원인 추적성과 운영 관측성 향상

원본 공식 예제를 프로덕션 수준의 예제 코드로 끌어올린 리팩토링입니다. 재현성, 자원 관리, 파일 무결성, 설정 관리까지 대부분의 운영 환경 요소를 반영했습니다.

