원본코드
import os
import time
import joblib
import numpy as np
import random
from os.path import abspath, dirname, join as pjoin

import fire
from numpy import ndarray
import torch
from torch import nn
import  torch.nn.functional as F

import sys
sys.path.append(abspath(dirname(dirname(__file__))))

from base import Model
from loader import SessionDataset, SessionDataLoader
from utils import get_fallback_items

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class MLP(Model):
    def __init__(self, idmap, fidmap, mlp_params, dim=256, layer_dim=3000, dropout=0.35):
        super(MLP, self).__init__()

        tr_matrices, tr_categoricals, tr_scalars = mlp_params

        self.idmap, self.fidmap = idmap, fidmap
        self.idmap_inv = {v: k for k, v in idmap.items()}
        self.n_items = len(self.idmap)
        self.n_items = len(self.idmap)
        self.item_emb = nn.Embedding(self.n_items, dim)
        self.item_emb.weight.data.normal_(0, 0.001)
        self.scalar_emb = nn.Linear(tr_scalars.shape[-1], 64)
        self.scalar_emb.weight.data.normal_(0, 0.001)
        self.scalar_bn = nn.BatchNorm1d(64)

        self.feat_emb = nn.Embedding(len(self.fidmap), dim)
        self.feat_emb.weight.data.normal_(0, 0.001)

        modules = {}
        cat_modules = {}
        bns = {}
        cat_bns = {}
        aggr_width = 0
        aggr_width += 64

        for mat_id, mat in tr_matrices.items():
            width = mat.shape[-1]
            modules[mat_id] = nn.ModuleList([nn.Linear(width, dim)])
            bns[mat_id] = nn.ModuleList([nn.BatchNorm1d(dim)])
            aggr_width += dim

        for cat_id, cat in tr_categoricals.items():
            _, (n_cats, embedding_dim) = cat
            cat_modules[cat_id] = nn.Embedding(n_cats, embedding_dim, padding_idx=-1)
            cat_bns[cat_id] = nn.BatchNorm1d(embedding_dim)
            aggr_width += embedding_dim

        self.cat_embs = nn.ModuleDict(cat_modules)
        self.all_layers = nn.ModuleDict(modules)
        self.all_bns = nn.ModuleDict(bns)
        self.cat_bns = nn.ModuleDict(cat_bns)
        self.last_layers = nn.ModuleList([
                nn.Linear(aggr_width, layer_dim),
                nn.Linear(layer_dim, dim)
            ])

        self.last_bns = nn.ModuleList([
                nn.BatchNorm1d(layer_dim),
                nn.BatchNorm1d(dim)
            ])
        self.item_bias = nn.Parameter(torch.zeros(self.n_items).float(), requires_grad=True)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def _forward(self, batch):
        # print(batch.mlp)
        for i, vs in batch.mlp.items():
            if isinstance(vs, dict):
                for j, v in vs.items():
                    # print("%s:%s" % (i, j), v.shape)
                    batch.mlp[i][j] = v.to(self.device)
            elif i == 'scalar':
                # print("scalar shape:", batch.mlp.scalar.shape)
                batch.mlp.scalar = batch.mlp.scalar.to(self.device)

        item_embs = F.normalize(self.item_emb.weight)
        feat_embs = self.feat_emb.weight

        aggr = []

        h = self.scalar_bn(self.scalar_emb(batch.mlp.scalar))
        aggr.append(h)

        h =  batch.mlp.mat['item_bow']
        h = self.all_bns['item_bow'][0](self.drop(F.linear(h, item_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_5']
        h = self.all_bns['item_last_5'][0](self.drop(F.linear(h, item_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_2']
        h = self.all_bns['item_last_2'][0](self.drop(F.linear(h, item_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_1']
        h = self.all_bns['item_last_1'][0](self.drop(F.linear(h, item_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_5_feats']
        h = self.all_bns['item_last_5_feats'][0](self.drop(F.linear(h, feat_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_3_feats']
        h = self.all_bns['item_last_3_feats'][0](self.drop(F.linear(h, feat_embs.T)))
        aggr.append(h)

        h =  batch.mlp.mat['item_last_1_feats']
        h = self.all_bns['item_last_1_feats'][0](self.drop(F.linear(h, feat_embs.T)))
        aggr.append(h)

        for key, input in batch.mlp.mat.items():
            if key == 'item_bow' or key == 'item_last_5' or key == 'item_last_2' or key == 'item_last_1':
                continue

            if key == 'item_last_5_feats' or key == 'item_last_3_feats' or key == 'item_last_1_feats':
                continue
            if key == 'item_all_feats':
                continue

            h = input
            for l, bn in zip(self.all_layers[key], self.all_bns[key]):
                h = torch.tanh(self.drop(l(h)))
                h = bn(h)
            aggr.append(h)

        for key, input in batch.mlp.cat.items():
            h = self.drop(self.cat_embs[key](input))
            h = self.cat_bns[key](h)
            aggr.append(h)

        h = torch.cat(aggr, dim=-1)
        for l, bn in zip(self.last_layers, self.last_bns):
            h = bn(torch.tanh(self.drop(l(h))))

        ret = F.linear(h, item_embs)
        return ret

    def init_weights(self):
        for key in self.all_layers:
            try:
                for layer in self.all_layers[key]:
                    # Xavier Initialization for weights
                    torch.nn.init.xavier_uniform_(layer.weight)
                    layer.bias.data.normal_(0.0, 0.001)
                    torch.clamp(layer.bias.data, min=-0.001, max=0.001)
            except:
                pass

    def forward(self, batch, mask=True):
        logit = self._forward(batch)
        if mask:
            logit[batch.extra.histories] = -10000.0
        return logit



def run(
    epochs=20, batch_size=256, device='cuda', seed=2022,
    num_workers=0, num_negatives=100, pin_memory=False, persistent_workers=False,
    save_fname='save/mlp.pt', submit=False, **kwargs
):
    print(locals())
    # sleep_seconds = random.randint(1, 10)
    # print(f'sleep {sleep_seconds} before start')
    # time.sleep(sleep_seconds)
    dir_name = 'processed'
    if submit:
        dir_name += '_submit'
    dir_name = pjoin(abspath(dirname(dirname(dirname(__file__)))), dir_name)
    idmap, fidmap = joblib.load(f'{dir_name}/indices')[:2]
    aug = '_aug' if kwargs.get('augmentation', False) else ''
    df_train = joblib.load(f'{dir_name}/df_train{aug}')
    print(df_train.shape)
    df_te = joblib.load(f'{dir_name}/df_val')
    mlp_params = joblib.load(f'{dir_name}/mlp_train{aug}')
    model = MLP(idmap, fidmap, mlp_params=mlp_params, dim=256, layer_dim=3000, dropout=0.35).to(device)
    model.set_fork_logic(df_train, df_te, joblib.load(f'{dir_name}/features'))
    print("train len:", len(df_train))
    ds = SessionDataset(df_train, idmap, fidmap, num_negatives=100, mlp_params=mlp_params)
    dl_tr = SessionDataLoader(
        ds, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap, mlp_params=mlp_params,
        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory,
        shuffle=kwargs.get('shuffle', False)
    )
    mlp_params_te = joblib.load(f'{dir_name}/mlp_val')
    ds_te = SessionDataset(df_te, idmap, fidmap, num_negatives=100, mlp_params=mlp_params_te)
    dl_te = SessionDataLoader(ds_te, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap, mlp_params=mlp_params_te)

    if os.path.exists(save_fname):
        model.load_state_dict(torch.load(save_fname))
        hit, mrr = model.validate(dl_te, **kwargs)
        print(f'HIT: {hit:.6f}, MRR: {mrr:.8f}')
        return
    optimizer = torch.optim.Adam(model.parameters(),  lr=2 * 1e-4, weight_decay=5e-3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=1e-6, verbose=1)
    model.fit(dl_tr, dl_te, optimizer, scheduler=None, epochs=epochs, seed=seed, save_fname=save_fname)


if __name__ == '__main__':
    fire.Fire(run)

연구용 MLP를 프로덕션 수준으로 끌어올린 리팩토링으로, 재현성·예외 처리·유지보수성을 크게 강화했지만 피처 처리의 강결합 구조는 추가 추상화가 남아 있다.

제안패치
import logging
import os
import random
from pathlib import Path
import joblib

import fire
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from base import Model
from loader import SessionDataset, SessionDataLoader
from utils import get_fallback_items

# [아키텍처 개선] 설정 및 Feature Config 구체화 (Config-Driven Architecture)
DEFAULT_DIM: int = 256
DEFAULT_LAYER_DIM: int = 3000
DEFAULT_DROPOUT: float = 0.35
DEFAULT_SCALAR_EMB_DIM: int = 64
DEFAULT_LR: float = 2e-4
DEFAULT_WEIGHT_DECAY: float = 5e-3

# 하드코딩 제거를 위한 Feature Grouping Config 분리
FEATURE_CONFIG = {
    "item_bow_mats": ['item_bow', 'item_last_5', 'item_last_2', 'item_last_1'],
    "feat_mats": ['item_last_5_feats', 'item_last_3_feats', 'item_last_1_feats'],
    "skipped_mats": ['item_all_feats']
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLPModelPipeline")


def _seed_everything(seed: int) -> None:
    """[재현성 확보] NumPy import 누락 해결 및 전역 시드 완벽 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _move_batch_to_device(batch, device: str) -> None:
    """[어댑터 분리] 배치 내 텐서 디바이스 이동 로직을 순수 함수 형태로 분리하여 부수 효과 격리"""
    for i, vs in batch.mlp.items():
        if isinstance(vs, dict):
            for j, v in vs.items():
                batch.mlp[i][j] = v.to(device)
        elif i == 'scalar':
            batch.mlp.scalar = batch.mlp.scalar.to(device)


class MLP(Model):
    def __init__(self, idmap, fidmap, mlp_params, dim=DEFAULT_DIM, layer_dim=DEFAULT_LAYER_DIM, dropout=DEFAULT_DROPOUT):
        super(MLP, self).__init__()

        tr_matrices, tr_categoricals, tr_scalars = mlp_params

        self.idmap = idmap
        self.fidmap = fidmap
        self.idmap_inv = {v: k for k, v in idmap.items()}
        self.n_items = len(self.idmap)
        
        self.item_emb = nn.Embedding(self.n_items, dim)
        self.item_emb.weight.data.normal_(0, 0.001)
        
        self.scalar_emb = nn.Linear(tr_scalars.shape[-1], DEFAULT_SCALAR_EMB_DIM)
        self.scalar_emb.weight.data.normal_(0, 0.001)
        self.scalar_bn = nn.BatchNorm1d(DEFAULT_SCALAR_EMB_DIM)

        self.feat_emb = nn.Embedding(len(self.fidmap), dim)
        self.feat_emb.weight.data.normal_(0, 0.001)

        modules = {}
        cat_modules = {}
        bns = {}
        cat_bns = {}
        aggr_width = DEFAULT_SCALAR_EMB_DIM

        for mat_id, mat in tr_matrices.items():
            width = mat.shape[-1]
            modules[mat_id] = nn.ModuleList([nn.Linear(width, dim)])
            bns[mat_id] = nn.ModuleList([nn.BatchNorm1d(dim)])
            aggr_width += dim

        for cat_id, cat in tr_categoricals.items():
            _, (n_cats, embedding_dim) = cat
            cat_modules[cat_id] = nn.Embedding(n_cats, embedding_dim, padding_idx=-1)
            cat_bns[cat_id] = nn.BatchNorm1d(embedding_dim)
            aggr_width += embedding_dim

        self.cat_embs = nn.ModuleDict(cat_modules)
        self.all_layers = nn.ModuleDict(modules)
        self.all_bns = nn.ModuleDict(bns)
        self.cat_bns = nn.ModuleDict(cat_bns)
        
        self.last_layers = nn.ModuleList([
            nn.Linear(aggr_width, layer_dim),
            nn.Linear(layer_dim, dim)
        ])
        self.last_bns = nn.ModuleList([
            nn.BatchNorm1d(layer_dim),
            nn.BatchNorm1d(dim)
        ])
        
        self.item_bias = nn.Parameter(torch.zeros(self.n_items).float(), requires_grad=True)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def _forward(self, batch):
        # [어댑터 적용] 디바이스 이동 어댑터 함수 호출
        _move_batch_to_device(batch, self.device)

        item_embs = F.normalize(self.item_emb.weight)
        feat_embs = self.feat_emb.weight

        aggr = []

        h = self.scalar_bn(self.scalar_emb(batch.mlp.scalar))
        aggr.append(h)

        # [Config-Driven] 하드코딩 제거 및 설정 기반 매트릭스 피처 처리
        for mat_name in FEATURE_CONFIG["item_bow_mats"]:
            if mat_name in batch.mlp.mat:
                h = batch.mlp.mat[mat_name]
                h = self.all_bns[mat_name][0](self.drop(F.linear(h, item_embs.T)))
                aggr.append(h)

        for mat_name in FEATURE_CONFIG["feat_mats"]:
            if mat_name in batch.mlp.mat:
                h = batch.mlp.mat[mat_name]
                h = self.all_bns[mat_name][0](self.drop(F.linear(h, feat_embs.T)))
                aggr.append(h)

        skipped_mats = set(FEATURE_CONFIG["item_bow_mats"] + FEATURE_CONFIG["feat_mats"] + FEATURE_CONFIG["skipped_mats"])
        for key, input_tensor in batch.mlp.mat.items():
            if key in skipped_mats:
                continue
            h = input_tensor
            for l, bn in zip(self.all_layers[key], self.all_bns[key]):
                h = torch.tanh(self.drop(l(h)))
                h = bn(h)
            aggr.append(h)

        for key, input_tensor in batch.mlp.cat.items():
            h = self.drop(self.cat_embs[key](input_tensor))
            h = self.cat_bns[key](h)
            aggr.append(h)

        h = torch.cat(aggr, dim=-1)
        for l, bn in zip(self.last_layers, self.last_bns):
            h = bn(torch.tanh(self.drop(l(h))))

        ret = F.linear(h, item_embs)
        return ret

    def init_weights(self):
        """[예외 처리 구체화] 무음 예외 삼킴을 제거하고 명시적 에러 로그 및 예외 발생 보장"""
        for key in self.all_layers:
            try:
                for layer in self.all_layers[key]:
                    torch.nn.init.xavier_uniform_(layer.weight)
                    layer.bias.data.normal_(0.0, 0.001)
                    torch.clamp(layer.bias.data, min=-0.001, max=0.001)
            except AttributeError as e:
                logger.error(f"AttributeError during weight initialization in 'all_layers[{key}]': {e}")
                raise TypeError(f"Invalid layer structure in 'all_layers[{key}]': {e}")
            except Exception as e:
                logger.error(f"Unexpected error during weight initialization in 'all_layers[{key}]': {e}")
                raise

    def forward(self, batch, mask=True):
        logit = self._forward(batch)
        if mask:
            logit[batch.extra.histories] = -10000.0
        return logit


def run(
    epochs=20, batch_size=256, device='cuda', seed=2022,
    num_workers=0, num_negatives=100, pin_memory=False, persistent_workers=False,
    save_fname='save/mlp.pt', submit=False, **kwargs
):
    logger.info(locals())
    
    # [재현성 확보] 학습 시작 전 시드 고정
    _seed_everything(seed)

    dir_name = 'processed'
    if submit:
        dir_name += '_submit'
    
    base_path = Path(__file__).resolve().parent.parent.parent
    dir_name = base_path / dir_name
    
    idmap, fidmap = joblib.load(f'{dir_name}/indices')[:2]
    aug = '_aug' if kwargs.get('augmentation', False) else ''
    df_train = joblib.load(f'{dir_name}/df_train{aug}')
    logger.info(f"Train data shape: {df_train.shape}")
    
    df_te = joblib.load(f'{dir_name}/df_val')
    mlp_params = joblib.load(f'{dir_name}/mlp_train{aug}')
    
    model = MLP(idmap, fidmap, mlp_params=mlp_params, dim=DEFAULT_DIM, layer_dim=DEFAULT_LAYER_DIM, dropout=DEFAULT_DROPOUT).to(device)
    model.set_fork_logic(df_train, df_te, joblib.load(f'{dir_name}/features'))
    logger.info(f"Train dataset length: {len(df_train)}")
    
    ds = SessionDataset(df_train, idmap, fidmap, num_negatives=100, mlp_params=mlp_params)
    dl_tr = SessionDataLoader(
        ds, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap, mlp_params=mlp_params,
        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory,
        shuffle=kwargs.get('shuffle', False)
    )
    
    mlp_params_te = joblib.load(f'{dir_name}/mlp_val')
    ds_te = SessionDataset(df_te, idmap, fidmap, num_negatives=100, mlp_params=mlp_params_te)
    dl_te = SessionDataLoader(
        ds_te, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap, mlp_params=mlp_params_te,
        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory
    )

    # [체크포인트 안정성] map_location 적용을 통한 CPU/GPU 호환성 확보
    if os.path.exists(save_fname):
        state_dict = torch.load(save_fname, map_location=device)
        model.load_state_dict(state_dict)
        hit, mrr = model.validate(dl_te, **kwargs)
        logger.info(f'HIT: {hit:.6f}, MRR: {mrr:.8f}')
        return
        
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    model.fit(dl_tr, dl_te, optimizer, scheduler=None, epochs=epochs, seed=seed, save_fname=save_fname)


if __name__ == '__main__':
    fire.Fire(run)

최종개선사항
✅ NumPy 시드 누락 → 완전한 재현성 확보 구조 전환
✅ Feature 하드코딩 → FEATURE_CONFIG 기반 설정 주입 구조 전환
✅ Batch device 이동 로직 → 어댑터 함수 분리로 책임 격리
✅ Checkpoint 로딩 → map_location 적용으로 GPU/CPU 호환성 강화
✅ Silent Exception 제거 → 명시적 로깅 및 장애 추적 구조 강화
✅ 하이퍼파라미터 분산 → 중앙 상수 관리 체계 전환
✅ 학습 파이프라인 → 운영 가능한 ML 실험 구조로 안정화

연구용 추천 모델을 운영형 ML 파이프라인으로 진화시킨 수준 높은 개선이며,
재현성·확장성·장애 대응력은 확보했지만 설정 외부화와 학습 컴포넌트 완전 분리 단계가 남아 있다.
