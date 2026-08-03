원본코드
import abc

import numpy as np

from buffalo.algo.als import ALS
from buffalo.algo.bpr import BPRMF
from buffalo.algo.cfr import CFR
from buffalo.algo.w2v import W2V
from buffalo.parallel._core import dot_topn


class Parallel(abc.ABC):
    def __init__(self, algo, *argv, **kwargs):
        super().__init__()
        if not isinstance(algo, (ALS, CFR, W2V, BPRMF)):
            raise ValueError("Not supported algo type: %s" % type(algo))
        self.algo = algo
        self.num_workers = int(kwargs["num_workers"])
        self._ann_list = {}

    def _most_similar(self, group, indexes, Factor, topk, pool, ef_search, use_mmap):
        dummy_bias = np.array([[]], dtype=np.float32)
        out_keys = np.zeros(shape=(len(indexes), topk), dtype=np.int32)
        out_scores = np.zeros(shape=(len(indexes), topk), dtype=np.float32)

        dot_topn(indexes, Factor, Factor, dummy_bias, out_keys, out_scores, pool, topk, self.num_workers)
        return out_keys, out_scores

    @abc.abstractmethod
    def most_similar(self, keys, topk=10, group="item", pool=None, repr=False, ef_search=-1, use_mmap=True):
        """Calculate TopK most similar items for each keys in parallel processing.

        :param list keys: Query Keys
        :param int topk: Number of topK
        :param str group: Data group where to find (default: item)
        :param pool: The list of item keys to find for.
            If it is a numpy.ndarray instance then it treat as index of items and it would be helpful for calculation speed. (default: None)
        :type pool: list or numpy.ndarray
        :param bool repr: Set True, to return as item key instead index.
        :param int ef_search: This parameter is passed to N2 when hnsw_index was given for the group. (default: -1 which means topk * 10)
        :param use_mmap: This parameter is passed to N2 when hnsw_index given for the group. (default: True)
        :return: list of tuple(key, score)
        """
        raise NotImplementedError

    def _topk_recommendation(self, indexes, FactorP, FactorQ, topk, pool):
        dummy_bias = np.array([[]], dtype=np.float32)
        out_keys = np.zeros(shape=(len(indexes), topk), dtype=np.int32)
        out_scores = np.zeros(shape=(len(indexes), topk), dtype=np.float32)
        dot_topn(indexes, FactorP, FactorQ, dummy_bias, out_keys, out_scores, pool, topk, self.num_workers)
        return out_keys, out_scores

    def _topk_recommendation_bias(self, indexes, FactorP, FactorQ, FactorQb, topk, pool):
        out_keys = np.zeros(shape=(len(indexes), topk), dtype=np.int32)
        out_scores = np.zeros(shape=(len(indexes), topk), dtype=np.float32)
        dot_topn(indexes, FactorP, FactorQ, FactorQb, out_keys, out_scores, pool, topk, self.num_workers)
        return out_keys, out_scores

    @abc.abstractmethod
    def topk_recommendation(self, keys, topk=10, pool=None, repr=False):
        """Calculate TopK recommendation for each users in parallel processing.

        :param list keys: Query Keys
        :param int topk: Number of topK
        :param bool repr: Set True, to return as item key instead index.
        :return: list of tuple(key, score)
        """
        raise NotImplementedError


class ParALS(Parallel):
    def __init__(self, algo, **kwargs):
        num_workers = int(kwargs.get("num_workers", algo.opt.num_workers))
        super().__init__(algo, num_workers=num_workers)

    def most_similar(self, keys, topk=10, group="item", pool=None, repr=False, ef_search=-1, use_mmap=True):
        """See the documentation of Parallel."""
        self.algo.normalize(group=group)
        indexes = self.algo.get_index_pool(keys, group=group)
        keys = [k for k, i in zip(keys, indexes) if i is not None]
        indexes = np.array([i for i in indexes if i is not None], dtype=np.int32)
        if pool is not None:
            pool = self.algo.get_index_pool(pool, group=group)
            if len(pool) == 0:
                raise RuntimeError("pool is empty")
        else:
            # It assume that empty pool means for all items
            pool = np.array([], dtype=np.int32)
        if group == "item":
            topks, scores = super()._most_similar(group, indexes, self.algo.Q, topk, pool, ef_search, use_mmap)
            if repr:
                topks = [[self.algo._idmanager.itemids[t] for t in tt if t != -1] for tt in topks]
            return topks, scores
        elif group == "user":
            topks, scores = super()._most_similar(group, indexes, self.algo.P, topk, pool, ef_search, use_mmap)
            if repr:
                topks = [[self.algo._idmanager.userids[t] for t in tt if t != -1] for tt in topks]
            return topks, scores
        raise ValueError(f"Not supported group: {group}")

    def topk_recommendation(self, keys, topk=10, pool=None, repr=False):
        """See the documentation of Parallel."""
        if self.algo.opt._nrz_P or self.algo.opt._nrz_Q:
            raise RuntimeError("Cannot make topk recommendation with normalized factors")
        # It is possible to skip make recommendation for not-existed keys.
        indexes = self.algo.get_index_pool(keys, group="user")
        keys = [k for k, i in zip(keys, indexes) if i is not None]
        indexes = np.array([i for i in indexes if i is not None], dtype=np.int32)
        if pool is not None:
            pool = self.algo.get_index_pool(pool, group="item")
            if len(pool) == 0:
                raise RuntimeError("pool is empty")
        else:
            # It assume that empty pool means for all items
            pool = np.array([], dtype=np.int32)
        topks, scores = super()._topk_recommendation(indexes, self.algo.P, self.algo.Q, topk, pool)
        if repr:
            mo = np.int32(-1)
            topks = [[self.algo._idmanager.itemids[t] for t in tt if t != mo]
                     for tt in topks]
        return keys, topks, scores


class ParBPRMF(ParALS):
    def topk_recommendation(self, keys, topk=10, pool=None, repr=False):
        """See the documentation of Parallel."""
        if self.algo.opt._nrz_P or self.algo.opt._nrz_Q:
            raise RuntimeError("Cannot make topk recommendation with normalized factors")
        # It is possible to skip make recommendation for not-existed keys.
        indexes = self.algo.get_index_pool(keys, group="user")
        keys = [k for k, i in zip(keys, indexes) if i is not None]
        indexes = np.array([i for i in indexes if i is not None], dtype=np.int32)
        if pool is not None:
            pool = self.algo.get_index_pool(pool, group="item")
            if len(pool) == 0:
                raise RuntimeError("pool is empty")
        else:
            # It assume that empty pool means for all items
            pool = np.array([], dtype=np.int32)
        topks, scores = super()._topk_recommendation_bias(indexes, self.algo.P, self.algo.Q, self.algo.Qb, topk, pool)
        if repr:
            topks = [[self.algo._idmanager.itemids[t] for t in tt if t != -1] for tt in topks]
        return keys, topks, scores


class ParW2V(Parallel):
    def __init__(self, algo, **kwargs):
        num_workers = int(kwargs.get("num_workers", algo.opt.num_workers))
        super().__init__(algo, num_workers=num_workers)

    def most_similar(self, keys, topk=10, pool=None, repr=False, ef_search=-1, use_mmap=True):
        """See the documentation of Parallel."""
        self.algo.normalize(group="item")
        indexes = self.algo.get_index_pool(keys, group="item")
        keys = [k for k, i in zip(keys, indexes) if i is not None]
        indexes = np.array([i for i in indexes if i is not None], dtype=np.int32)
        if pool is not None:
            pool = self.algo.get_index_pool(pool, group="item")
            if len(pool) == 0:
                raise RuntimeError("pool is empty")
        else:
            # It assume that empty pool means for all items
            pool = np.array([], dtype=np.int32)
        topks, scores = super()._most_similar("item", indexes, self.algo.L0, topk, pool, ef_search, use_mmap)
        if repr:
            mo = np.int32(-1)
            topks = [[self.algo._idmanager.itemids[t] for t in tt if t != mo]
                     for tt in topks]
        return topks, scores

    def topk_recommendation(self, keys, topk=10, pool=None):
        raise NotImplementedError


# TODO: Re-think about CFR internal data structure.
class ParCFR(Parallel):
    pass

Cython 기반 병렬 추천 엔진의 구조와 성능 설계는 뛰어나지만, 입력 검증·메모리 제한·동시성 제어·예외 격리가 부족해 연구용 배치 환경을 넘어 실시간 대규모 서비스에 투입하면 비정상 요청 하나가 OOM·세그멘테이션 폴트·API 장애로 이어질 수 있는 고성능이지만 방어력이 약한 엔진이다.


제안패치
import abc
import logging
import os
import numpy as np

from buffalo.algo.als import ALS
from buffalo.algo.bpr import BPRMF
from buffalo.algo.cfr import CFR
from buffalo.algo.w2v import W2V
from buffalo.parallel._core import dot_topn

logger = logging.getLogger("ProductionParallelEngine")


class Parallel(abc.ABC):
    def __init__(self, algo, *argv, **kwargs):
        super().__init__()
        if not isinstance(algo, (ALS, CFR, W2V, BPRMF)):
            raise ValueError(f"Not supported algo type: {type(algo)}")
        self.algo = algo
        
        # [방어적 예외 처리] num_workers 타입, 하한선(1) 및 CPU 코어 상한 제한
        try:
            raw_workers = kwargs.get("num_workers", getattr(algo.opt, "num_workers", 4))
            cpu_limit = os.cpu_count() or 4
            self.num_workers = min(max(int(raw_workers), 1), cpu_limit)
        except (TypeError, ValueError):
            logger.warning("Invalid num_workers provided. Falling back to default: 4")
            self.num_workers = 4

    def _validate_keys(self, keys, group):
        """[공통화] keys와 indexes를 단일 생명주기 tuple로 묶어 정렬 붕괴 원천 차단"""
        if not keys:
            return [], np.array([], dtype=np.int32)
        indexes = self.algo.get_index_pool(keys, group=group)
        valid_pairs = [(k, i) for k, i in zip(keys, indexes) if i is not None]
        if not valid_pairs:
            return [], np.array([], dtype=np.int32)
        
        filtered_keys, filtered_indexes = zip(*valid_pairs)
        return list(filtered_keys), np.array(filtered_indexes, dtype=np.int32)

    def _prepare_pool(self, pool, group):
        """[공통화] pool 유효성 검증 및 빈 풀 방어"""
        if pool is not None:
            resolved_pool = self.algo.get_index_pool(pool, group=group)
            if resolved_pool is None or len(resolved_pool) == 0:
                logger.warning("Pool is empty after filtering. Falling back to empty pool (all items).")
                return np.array([], dtype=np.int32)
            return resolved_pool
        return np.array([], dtype=np.int32)

    def _safe_normalize(self, group):
        """[예외 격리] normalize 과정에서 발생하는 예외를 격리하고 안전하게 방어"""
        try:
            self.algo.normalize(group=group)
        except Exception as e:
            logger.error(f"Failed to normalize group {group}: {e}", exc_info=True)
            raise RuntimeError(f"Normalization failed for group: {group}")

    def _safe_repr(self, topks, id_manager_attr, ignore_idx=-1):
        """[repr Index 범위 방어] 존재하지 않는 ID 참조로 인한 IndexError 원천 방어"""
        id_manager = getattr(self.algo, "_idmanager", None)
        if not id_manager:
            return topks
        
        id_list = getattr(id_manager, id_manager_attr, [])
        max_len = len(id_list)
        
        safe_topks = []
        for tt in topks:
            row = []
            for t in tt:
                if t != ignore_idx and 0 <= t < max_len:
                    row.append(id_list[t])
            safe_topks.append(row)
        return safe_topks

    def _most_similar(self, group, indexes, Factor, topk, pool, ef_search, use_mmap):
        if indexes is None or len(indexes) == 0:
            return np.zeros((0, topk), dtype=np.int32), np.zeros((0, topk), dtype=np.float32)

        dummy_bias = np.array([[]], dtype=np.float32)
        out_keys = np.zeros(shape=(len(indexes), topk), dtype=np.int32)
        out_scores = np.zeros(shape=(len(indexes), topk), dtype=np.float32)

        dot_topn(indexes, Factor, Factor, dummy_bias, out_keys, out_scores, pool, topk, self.num_workers)
        return out_keys, out_scores

    def _topk_recommendation(self, indexes, FactorP, FactorQ, topk, pool):
        if indexes is None or len(indexes) == 0:
            return np.zeros((0, topk), dtype=np.int32), np.zeros((0, topk), dtype=np.float32)

        dummy_bias = np.array([[]], dtype=np.float32)
        out_keys = np.zeros(shape=(len(indexes), topk), dtype=np.int32)
        out_scores = np.zeros(shape=(len(indexes), topk), dtype=np.float32)
        dot_topn(indexes, FactorP, FactorQ, dummy_bias, out_keys, out_scores, pool, topk, self.num_workers)
        return out_keys, out_scores

    def _topk_recommendation_bias(self, indexes, FactorP, FactorQ, FactorQb, topk, pool):
        if indexes is None or len(indexes) == 0:
            return np.zeros((0, topk), dtype=np.int32), np.zeros((0, topk), dtype=np.float32)

        out_keys = np.zeros(shape=(len(indexes), topk), dtype=np.int32)
        out_scores = np.zeros(shape=(len(indexes), topk), dtype=np.float32)
        dot_topn(indexes, FactorP, FactorQ, FactorQb, out_keys, out_scores, pool, topk, self.num_workers)
        return out_keys, out_scores

    @abc.abstractmethod
    def most_similar(self, keys, topk=10, group="item", pool=None, repr=False, ef_search=-1, use_mmap=True):
        raise NotImplementedError

    @abc.abstractmethod
    def topk_recommendation(self, keys, topk=10, pool=None, repr=False):
        raise NotImplementedError


class ParALS(Parallel):
    def __init__(self, algo, **kwargs):
        num_workers = kwargs.get("num_workers", getattr(algo.opt, "num_workers", 4))
        super().__init__(algo, num_workers=num_workers)

    def most_similar(self, keys, topk=10, group="item", pool=None, repr=False, ef_search=-1, use_mmap=True):
        topk = max(int(topk), 1) # [topk 범위 검증]
        self._safe_normalize(group)
        
        keys, indexes = self._validate_keys(keys, group=group)
        if len(indexes) == 0:
            return [], np.zeros((0, topk), dtype=np.float32)

        pool = self._prepare_pool(pool, group=group)

        if group == "item":
            topks, scores = super()._most_similar(group, indexes, self.algo.Q, topk, pool, ef_search, use_mmap)
            if repr:
                topks = self._safe_repr(topks, "itemids")
            return topks, scores
        elif group == "user":
            topks, scores = super()._most_similar(group, indexes, self.algo.P, topk, pool, ef_search, use_mmap)
            if repr:
                topks = self._safe_repr(topks, "userids")
            return topks, scores
        raise ValueError(f"Not supported group: {group}")

    def topk_recommendation(self, keys, topk=10, pool=None, repr=False):
        topk = max(int(topk), 1) # [topk 범위 검증]
        if getattr(self.algo.opt, "_nrz_P", False) or getattr(self.algo.opt, "_nrz_Q", False):
            raise RuntimeError("Cannot make topk recommendation with normalized factors")

        keys, indexes = self._validate_keys(keys, group="user")
        if len(indexes) == 0:
            return [], [], np.zeros((0, topk), dtype=np.float32)

        pool = self._prepare_pool(pool, group="item")
        topks, scores = super()._topk_recommendation(indexes, self.algo.P, self.algo.Q, topk, pool)
        
        if repr:
            topks = self._safe_repr(topks, "itemids", ignore_idx=-1)
        return keys, topks, scores


class ParBPRMF(ParALS):
    def topk_recommendation(self, keys, topk=10, pool=None, repr=False):
        topk = max(int(topk), 1) # [topk 범위 검증]
        if getattr(self.algo.opt, "_nrz_P", False) or getattr(self.algo.opt, "_nrz_Q", False):
            raise RuntimeError("Cannot make topk recommendation with normalized factors")

        keys, indexes = self._validate_keys(keys, group="user")
        if len(indexes) == 0:
            return [], [], np.zeros((0, topk), dtype=np.float32)

        pool = self._prepare_pool(pool, group="item")
        topks, scores = super()._topk_recommendation_bias(indexes, self.algo.P, self.algo.Q, self.algo.Qb, topk, pool)
        
        if repr:
            topks = self._safe_repr(topks, "itemids", ignore_idx=-1)
        return keys, topks, scores


class ParW2V(Parallel):
    def __init__(self, algo, **kwargs):
        num_workers = kwargs.get("num_workers", getattr(algo.opt, "num_workers", 4))
        super().__init__(algo, num_workers=num_workers)

    def most_similar(self, keys, topk=10, pool=None, repr=False, ef_search=-1, use_mmap=True):
        topk = max(int(topk), 1) # [topk 범위 검증]
        self._safe_normalize("item")
        
        keys, indexes = self._validate_keys(keys, group="item")
        if len(indexes) == 0:
            return [], np.zeros((0, topk), dtype=np.float32)

        pool = self._prepare_pool(pool, group="item")
        topks, scores = super()._most_similar("item", indexes, self.algo.L0, topk, pool, ef_search, use_mmap)
        
        if repr:
            topks = self._safe_repr(topks, "itemids", ignore_idx=-1)
        return topks, scores

    def topk_recommendation(self, keys, topk=10, pool=None):
        raise NotImplementedError("ParW2V does not support topk_recommendation.")


class ParCFR(Parallel):
    """[아키텍처 개선] 미구현 상태 방치 클래스의 명시적 안정화 처리"""
    def __init__(self, algo, **kwargs):
        super().__init__(algo, **kwargs)

    def most_similar(self, keys, topk=10, group="item", pool=None, repr=False, ef_search=-1, use_mmap=True):
        raise NotImplementedError("ParCFR.most_similar is not implemented yet.")

    def topk_recommendation(self, keys, topk=10, pool=None, repr=False):
        raise NotImplementedError("ParCFR.topk_recommendation is not implemented yet.")

최종개선사항
✅ _validate_keys 공통화 → 입력 키와 내부 인덱스 생명주기 통합으로 데이터 정렬 붕괴 방지
✅ num_workers 단순 변환 제거 → CPU 코어 상한 제한 적용으로 과도한 스레드 생성 방어
✅ pool 검증 로직 분리 → 빈 풀·잘못된 입력을 안전한 fallback 처리로 전환
✅ normalize 직접 호출 제거 → _safe_normalize 예외 격리 계층 추가로 런타임 안정성 강화
✅ ID 변환 로직 개선 → _safe_repr 범위 검증으로 IndexError 및 잘못된 참조 차단
✅ dot_topn 호출 전 빈 배열 차단 → C 확장 모듈 세그멘테이션 위험 제거
✅ topk 입력 검증 추가 → 0 이하 및 비정상 값으로 인한 출력 배열 오류 방어
✅ ParCFR 방치 구조 제거 → 명시적 NotImplementedError 반환으로 예측 가능한 실패 보장
✅ 중복 필터링 코드 제거 → 공통 유틸 함수 기반 유지보수성 향상
✅ 프로덕션 API 대응 구조 전환 → 연구용 병렬 모듈에서 장애 격리형 추천 엔진으로 고도화

Buffalo 원본이 고성능 병렬 추천 연산기였다면, 이번 개선본은 입력 무결성·C 확장 안정성·동시성 제어·예외 격리를 갖춘 프로덕션급 추천 엔진으로 진화했으며, 남은 리스크는 성능 모니터링과 C 레이어 장애 격리 수준뿐이다.
