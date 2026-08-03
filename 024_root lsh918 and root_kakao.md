원본코드
import json
import time

from buffalo.algo import ALS, ALSOption
from buffalo.data import MatrixMarketOptions
from buffalo.misc import aux, log
from buffalo.parallel import ParALS


def example1():
    log.set_log_level(log.DEBUG)
    als_option = ALSOption().get_default_option()
    als_option.validation = aux.Option({"topk": 10})
    data_option = MatrixMarketOptions().get_default_option()
    data_option.input.main = "../tests/ext/ml-100k/main"
    data_option.input.iid = "../tests/ext/ml-100k/iid"

    als = ALS(als_option, data_opt=data_option)
    als.initialize()
    als.train()
    print("MovieLens 100k metrics for validations\n%s" % json.dumps(als.get_validation_results(), indent=2))

    print("Similar movies to Star_Wars_(1977)")
    for rank, (movie_name, score) in enumerate(als.most_similar("49.Star_Wars_(1977)")):
        print(f"{rank + 1:02d}. {score:.3f} {movie_name}")


def example2():
    log.set_log_level(log.INFO)
    als_option = ALSOption().get_default_option()
    data_option = MatrixMarketOptions().get_default_option()
    data_option.input.main = "../tests/ext/ml-100k/main"
    data_option.input.iid = "../tests/ext/ml-100k/iid"
    # data_option.data.path = "./ml20m.h5py"
    data_option.data.use_cache = True

    als = ALS(als_option, data_opt=data_option)
    als.initialize()
    als.train()
    als.normalize("item")
    als.build_itemid_map()

    print("Make item recommendation on als.ml20m.par.top10.tsv with Parallel(Thread=4)")
    par = ParALS(als)
    par.num_workers = 4
    all_items = als._idmanager.itemids
    start_t = time.time()
    with open("als.ml20m.par.top10.tsv", "w") as fout:
        for idx in range(0, len(all_items), 128):
            topks, _ = par.most_similar(all_items[idx:idx + 128], repr=True)
            for q, p in zip(all_items[idx:idx + 128], topks):
                fout.write("%s\t%s\n" % (q, "\t".join(p)))
    print("took: %.3f secs" % (time.time() - start_t))

    try:
        from n2 import HnswIndex

        index = HnswIndex(als.Q.shape[1])
        for f in als.Q:
            index.add_data(f)
        index.build(n_threads=4)
        index.save("ml20m.n2.index")
        index.unload()
        print("Make item recommendation on als.ml20m.par.top10.tsv with Ann(Thread=1)")
        par.set_hnsw_index("ml20m.n2.index", "item")
        par.num_workers = 4
        start_t = time.time()
        with open("als.ml20m.ann.top10.tsv", "w") as fout:
            for idx in range(0, len(all_items), 128):
                topks, _ = par.most_similar(all_items[idx:idx + 128], repr=True)
                for q, p in zip(all_items[idx:idx + 128], topks):
                    fout.write("%s\t%s\n" % (q, "\t".join(p)))
        print("took: %.3f secs" % (time.time() - start_t))
    except ImportError:
        print("n2 is not installed. skip it")

구조는 우수하지만, 경로 하드코딩과 자원·예외 처리가 미흡해 프로덕션 투입 전 안정성 보강이 필요한 예제 코드다.

제안패치
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

from buffalo.algo import ALS, ALSOption
from buffalo.data import MatrixMarketOptions
from buffalo.misc import aux, log
from buffalo.parallel import ParALS

# [아키텍처 개선] 매직 넘버 상수화 (Configuration/Constant 분리)
DEFAULT_BATCH_SIZE: int = 128
DEFAULT_NUM_WORKERS: int = 4
DEFAULT_TOP_K: int = 10

logger = log.get_logger("ExampleRunner")


def _validate_and_get_dataset_paths() -> Tuple[str, str]:
    """상대 경로 의존성 및 파일 존재 여부 방어적 검증 수행"""
    base_path = Path(__file__).resolve().parent
    main_path = base_path / "../tests/ext/ml-100k/main"
    iid_path = base_path / "../tests/ext/ml-100k/iid"

    if not main_path.exists() or not iid_path.exists():
        raise FileNotFoundError(
            f"Dataset paths do not exist. Checked main: {main_path}, iid: {iid_path}"
        )
    return str(main_path), str(iid_path)


def _write_recommendations(file_path: str, all_items: List[str], par: ParALS, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    """Context Manager(with)를 통한 안전한 파일 I/O 보장"""
    with open(file_path, "w", encoding="utf-8") as fout:
        for idx in range(0, len(all_items), batch_size):
            batch_items = all_items[idx:idx + batch_size]
            topks, _ = par.most_similar(batch_items, repr=True)
            for q, p in zip(batch_items, topks):
                fout.write("%s\t%s\n" % (q, "\t".join(p)))


# [아키텍처 개선] 함수 책임을 명확히 분리한 모듈형 파이프라인 함수들
def train_model() -> ALS:
    """모델 초기화 및 학습 수행"""
    als_option = ALSOption().get_default_option()
    main_path, iid_path = _validate_and_get_dataset_paths()
    data_option = MatrixMarketOptions().get_default_option()
    data_option.input.main = main_path
    data_option.input.iid = iid_path
    data_option.data.use_cache = True

    als = ALS(als_option, data_opt=data_option)
    als.initialize()
    als.train()
    als.normalize("item")
    als.build_itemid_map()
    return als


def recommend_parallel(als: ALS, output_path: str = "als.ml20m.par.top10.tsv") -> None:
    """병렬 처리 기반 추천 결과 생성"""
    logger.info(f"Make item recommendation on {output_path} with Parallel(Thread={DEFAULT_NUM_WORKERS})")
    par = ParALS(als)
    par.num_workers = DEFAULT_NUM_WORKERS
    all_items = als._idmanager.itemids
    
    start_t = time.time()
    _write_recommendations(output_path, all_items, par)
    logger.info("took: %.3f secs" % (time.time() - start_t))


def build_and_save_ann_index(als: ALS, index_path: str = "ml20m.n2.index") -> Optional[object]:
    """[원자적 저장 적용] ANN 인덱스 생성 및 파셜 붕괴 방지용 임시 파일 rename 기법 적용"""
    try:
        from n2 import HnswIndex
        index = HnswIndex(als.Q.shape[1])
        for f in als.Q:
            index.add_data(f)
        index.build(n_threads=DEFAULT_NUM_WORKERS)
        
        # 원자적 저장을 위한 임시 파일 경로 생성
        tmp_index_path = f"{index_path}.tmp"
        index.save(tmp_index_path)
        os.replace(tmp_index_path, index_path)
        
        return index
    except ImportError:
        logger.warning("n2 package is not installed. Skipping ANN indexing step.")
        return None
    except Exception as e:
        logger.error(f"Error building ANN index: {str(e)}")
        # 실패 시 잔여 임시 파일 정리
        if 'tmp_index_path' in locals() and os.path.exists(tmp_index_path):
            try:
                os.remove(tmp_index_path)
            except OSError:
                pass
        raise


def recommend_ann(als: ALS, index_path: str = "ml20m.n2.index", output_path: str = "als.ml20m.ann.top10.tsv") -> None:
    """ANN 인덱스 기반 고속 추천 수행"""
    index = None
    try:
        index = build_and_save_ann_index(als, index_path)
        if index is None:
            return

        logger.info(f"Make item recommendation on {output_path} with Ann(Thread={DEFAULT_NUM_WORKERS})")
        par = ParALS(als)
        par.set_hnsw_index(index_path, "item")
        par.num_workers = DEFAULT_NUM_WORKERS
        all_items = als._idmanager.itemids

        start_t = time.time()
        _write_recommendations(output_path, all_items, par)
        logger.info("took: %.3f secs" % (time.time() - start_t))

    finally:
        if index is not None:
            try:
                index.unload()
            except Exception as unload_err:
                logger.warning(f"Failed to cleanly unload HnswIndex: {unload_err}")


def example1():
    log.set_log_level(log.DEBUG)
    
    als_option = ALSOption().get_default_option()
    als_option.validation = aux.Option({"topk": DEFAULT_TOP_K})
    
    main_path, iid_path = _validate_and_get_dataset_paths()
    data_option = MatrixMarketOptions().get_default_option()
    data_option.input.main = main_path
    data_option.input.iid = iid_path

    als = ALS(als_option, data_opt=data_option)
    als.initialize()
    als.train()
    
    logger.info("MovieLens 100k metrics for validations\n%s" % json.dumps(als.get_validation_results(), indent=2))
    logger.info("Similar movies to Star_Wars_(1977)")
    
    for rank, (movie_name, score) in enumerate(als.most_similar("49.Star_Wars_(1977)")):
        logger.info(f"{rank + 1:02d}. {score:.3f} {movie_name}")


def example2():
    log.set_log_level(log.INFO)
    
    # [아키텍처 개선] 거대 단일 함수였던 example2를 명확한 파이프라인 단위로 분리하여 테스트 용이성 및 가독성 극대화
    als = train_model()
    recommend_parallel(als)
    recommend_ann(als)

최종 개선사항
✅ 매직 넘버 제거 → DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, DEFAULT_TOP_K 상수로 분리하여 설정 일관성 확보.
✅ 단일 거대 함수 분해 → train_model(), recommend_parallel(), build_and_save_ann_index(), recommend_ann()으로 책임을 명확히 분리(SRP 적용).
✅ 데이터셋 경로 검증 모듈화 → _validate_and_get_dataset_paths()로 파일 존재 여부를 사전 검증하여 실행 안정성 강화.
✅ 추천 결과 저장 공통화 → _write_recommendations()로 파일 I/O 중복을 제거하고 Context Manager로 자원 누수 방지.
✅ ANN 인덱스 저장 안정화 → 임시 파일 저장 후 os.replace()를 적용해 원자적(Atomic) 저장을 구현하고 손상 파일 생성 방지.
✅ ANN 실패 복구 강화 → 인덱스 생성 실패 시 임시 파일을 자동 정리하여 파일 시스템 무결성 확보.
✅ C++ 자원 해제 보장 → finally에서 index.unload()를 호출하여 예외 발생 여부와 관계없이 네이티브 메모리 누수 방지.
✅ 로깅 체계 통일 → logger.info(), warning(), error() 기반으로 운영 환경의 관측성과 디버깅 효율 향상.
✅ 프로덕션 아키텍처 수준 개선 → 학습·병렬 추천·ANN 구축·ANN 추천을 독립 파이프라인으로 분리해 테스트성과 확장성을 크게 향상.

예제 수준을 넘어 프로덕션 아키텍처에 가까운 구조로 발전했으며, 안정성·유지보수성·확장성을 모두 갖춘 완성도 높은 리팩토링이다.
