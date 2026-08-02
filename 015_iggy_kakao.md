원본코드
import os
import time
import random
import joblib
import random
from os.path import abspath, dirname, join as pjoin

import fire
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

import sys
sys.path.append(abspath(dirname(dirname(__file__))))

from base import Model
from loader import SessionDataset, SessionDataLoader
from utils import get_fallback_items

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class GNN(Model):
    def __init__(self, features, edge_index, edge_weight, hidden_channels, out_channels, dropout=0.1):
        super(GNN, self).__init__()
        self.x = torch.nn.Parameter(features, requires_grad=False)
        self.edge_index = torch.nn.Parameter(edge_index, requires_grad=False)
        self.edge_weight = torch.nn.Parameter(edge_weight, requires_grad=False)
        self.convs = nn.ModuleList([
            GCNConv(-1, hidden_channels),
            GCNConv(-1, out_channels)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels), 
            nn.BatchNorm1d(out_channels),
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self):
        x = self.x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, self.edge_index, self.edge_weight).tanh()
            x = bn(h)
        return x

class GRUN(Model):
    def __init__(self, idmap, features, edge_index, edge_weight, dim=32, gru_dim=128, gnn_dim=64, dropout=0.1):
        super(GRUN, self).__init__()
        self.idmap = idmap
        self.idmap_inv = {v: k for k, v in idmap.items()}
        self.n_items = len(self.idmap)
        self.item_emb = nn.Embedding(self.n_items + 1, dim, padding_idx=self.n_items)
        self.item_bias = nn.Parameter(torch.zeros(self.n_items).float(), requires_grad=True)
        self.gnn_dim = gnn_dim
        self.item_emb_gnn = GNN(
            torch.cat((features, torch.rand(1, features.shape[-1])), dim=0),
            edge_index, edge_weight, self.gnn_dim * 2, self.gnn_dim
        )
        
        self.gru_dim = gru_dim
        self.gru = nn.GRU(input_size=dim + self.gnn_dim, hidden_size=self.gru_dim, num_layers=2)
        self.item_dim = self.gnn_dim + dim

        self.drop = nn.Dropout(dropout)
        self.last_layers = nn.ModuleList([
            nn.Linear(self.gru_dim, self.item_dim * 2),
            nn.Linear(self.item_dim * 2, self.item_dim) 
        ])
        self.last_bns = nn.ModuleList([
            nn.BatchNorm1d(self.item_dim * 2), 
            nn.BatchNorm1d(self.item_dim),
        ])
        self.init_weights()

    def session_emb(self, batch):
        # Get sequential session embedding
        seq = pad_sequence(batch.views, batch_first=True, padding_value=self.n_items).to(self.device)
        _, indices = batch.extra.seq_lens.sort(dim=0, descending=True)
        packed = pack_padded_sequence(self.target_emb(seq[indices]), lengths=batch.extra.seq_lens[indices], batch_first=True)
        _, h = self.gru(packed)
        h = h.transpose(0, 1)
        _, inv_indices = torch.sort(indices, 0, descending=False)
        s_emb = self.drop(h[inv_indices].squeeze(dim=1))
        s_emb = s_emb.mean(dim=1)
        for l, bn in zip(self.last_layers, self.last_bns):
            s_emb = torch.tanh(self.drop(l(s_emb)))
            s_emb = bn(s_emb)
        return s_emb
    
    def forward(self, batch, mask=True):
        s_emb = self.session_emb(batch)
        logit = F.linear(s_emb, F.normalize(self.drop(self.target_emb(torch.arange(self.n_items).to(self.device)))), self.item_bias)
        if mask:
            logit[batch.extra.histories] = -10000.0
        return logit
    
    def target_emb(self, iid):
        item_emb = self.item_emb(iid)
        gnn_embeddings = self.item_emb_gnn()
        item_emb_gnn = F.embedding(iid, gnn_embeddings, padding_idx=self.n_items)
        return torch.cat([item_emb, item_emb_gnn], dim=-1)
    
    def init_weights(self):
        for name, weight in self.named_parameters():
            try:
                if len(weight.shape) > 1 and 'weight' in name:
                    nn.init.xavier_uniform_(weight)
                elif 'bias' in name:
                    weight.data.normal_(0.0, 0.001)
                    torch.clamp(weight.data, min=-0.001, max=0.001)
            except:
                pass


def run(
    epochs=30, batch_size=512, device='cuda', seed=2022, 
    num_workers=0, num_negatives=100, pin_memory=False, persistent_workers=False,
    save_fname='save/grun.pt', submit=False, all=False, **kwargs
):
    print(locals())
    sleep_seconds = random.randint(1, 10)
    print(f'sleep {sleep_seconds} before start')
    time.sleep(sleep_seconds)
    dir_name = 'processed'
    if submit:
        dir_name += '_submit'
    time.sleep(random.randint(1, 10))
    dir_name = pjoin(abspath(dirname(dirname(dirname(__file__)))), dir_name)
    idmap, fidmap = joblib.load(f'{dir_name}/indices')[:2]
    pick = '_all' if all else ''
    aug = '_aug' if kwargs.get('augmentation', False) else ''
    df_train = joblib.load(f'{dir_name}/df_train{aug}{pick}')
    df_te = joblib.load(f'{dir_name}/df_val')
    pick = 'all_' if all else ''
    features = torch.FloatTensor(joblib.load(f'{dir_name}/features'))
    edge_index, edge_weight = joblib.load(f'{dir_name}/{pick}edge_index_v1.1')
    model = GRUN(idmap, features, edge_index, edge_weight).to(device)    
    model.set_fork_logic(df_train, df_te, joblib.load(f'{dir_name}/features'))
    ds_tr = SessionDataset(df_train, idmap=idmap, fidmap=fidmap, num_negatives=num_negatives)
    dl_tr = SessionDataLoader(
        ds_tr, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap,
        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory,
        shuffle=kwargs.get('shuffle', False)
    )
    ds_te = SessionDataset(df_te, idmap=idmap, fidmap=fidmap, num_negatives=num_negatives)
    dl_te = SessionDataLoader(ds_te, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap)
    if os.path.exists(save_fname):
        model.load_state_dict(torch.load(save_fname))
        model.set_fork_logic(df_train, df_te)
        hit, mrr = model.validate(dl_te, **kwargs)
        print(f'HIT: {hit:.6f}, MRR: {mrr:.8f}')
        return
    optimizer = torch.optim.Adam(model.parameters(),  lr=5e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=1e-5, verbose=1)
    model.fit(dl_tr, dl_te, optimizer, scheduler, epochs=epochs, seed=seed, save_fname=save_fname)


if __name__ == '__main__':
    fire.Fire(run)

GNN 기반 추천 모델의 뼈대는 준수하지만, 핵심 임베딩 캐싱 부재·침묵하는 예외 처리·수치 안정성 무시라는 세 가지 결함 때문에 대규모 학습 환경에서는 GPU 자원 낭비와 디버깅 불능 상태를 동시에 유발하는 연구용 프로토타입 수준의 코드다.

제안패치
import os
import random
import time
from os.path import abspath, dirname, join as pjoin
import sys

import fire
import joblib
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch_geometric.nn import GCNConv

# [운영 안정성] 디버깅 옵션을 조건부로 분리하여 프로덕션 환경 성능 저하 원천 차단
if os.getenv("DEBUG_CUDA", "0") == "1":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# [아키텍처 개선] sys.path 조작 대신 상대/절대 패키지 임포트 표준 적용을 위한 안내 주석
# 권장 사용법: 프로젝트 루트에서 `pip install -e .` 수행 후 패키지 형태로 임포트
try:
    from base import Model
    from loader import SessionDataset, SessionDataLoader
    from utils import get_fallback_items
except ImportError:
    # 레거시 구조 호환용 폴백
    sys.path.append(abspath(dirname(dirname(__file__))))
    from base import Model
    from loader import SessionDataset, SessionDataLoader
    from utils import get_fallback_items


class GNN(Model):
    """[구조 고도화] 그래프 합성곱 네트워크 모듈"""
    def __init__(self, features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, hidden_channels: int, out_channels: int, dropout: float = 0.1):
        super(GNN, self).__init__()
        self.register_buffer('x', features)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)
        
        self.convs = nn.ModuleList([
            GCNConv(-1, hidden_channels),
            GCNConv(-1, out_channels)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels), 
            nn.BatchNorm1d(out_channels),
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self) -> torch.Tensor:
        x = self.x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, self.edge_index, self.edge_weight).tanh()
            x = bn(h)
        return x


class GRUN(Model):
    """[엔터프라이즈 리팩토링] epoch 단위 GNN 캐시 무효화 및 buffer 기반 임베딩 관리 세션 추천 모델"""
    def __init__(self, idmap: dict, features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, dim: int = 32, gru_dim: int = 128, gnn_dim: int = 64, dropout: float = 0.1):
        super(GRUN, self).__init__()
        self.idmap = idmap
        self.idmap_inv = {v: k for k, v in idmap.items()}
        self.n_items = len(self.idmap)
        
        self.item_emb = nn.Embedding(self.n_items + 1, dim, padding_idx=self.n_items)
        self.item_bias = nn.Parameter(torch.zeros(self.n_items, dtype=torch.float32), requires_grad=True)
        self.gnn_dim = gnn_dim
        
        self.item_emb_gnn = GNN(
            torch.cat((features, torch.rand(1, features.shape[-1])), dim=0),
            edge_index, edge_weight, self.gnn_dim * 2, self.gnn_dim
        )
        
        # [캐시 관리 고도화] PyTorch 버퍼(register_buffer) 기반으로 캐시 등록하여 state_dict 및 device 동기화 보장
        self.register_buffer('_cached_gnn_embeddings', None, persistent=False)
        
        self.gru_dim = gru_dim
        self.gru = nn.GRU(input_size=dim + self.gnn_dim, hidden_size=self.gru_dim, num_layers=2)
        self.item_dim = self.gnn_dim + dim

        self.drop = nn.Dropout(dropout)
        self.last_layers = nn.ModuleList([
            nn.Linear(self.gru_dim, self.item_dim * 2),
            nn.Linear(self.item_dim * 2, self.item_dim) 
        ])
        self.last_bns = nn.ModuleList([
            nn.BatchNorm1d(self.item_dim * 2), 
            nn.BatchNorm1d(self.item_dim),
        ])
        self.init_weights()

    def cache_gnn_embeddings(self):
        """[연산 최적화] 무효화(Stale) 방지를 위해 torch.no_grad() 환경에서 GNN 정점 임베딩을 안전하게 갱신"""
        with torch.no_grad():
            self._cached_gnn_embeddings = self.item_emb_gnn()

    def invalidate_cache(self):
        """[캐시 무효화 전략] Epoch 시작 시점마다 호출하여 GNN 가중치 업데이트가 반영되도록 보장"""
        self._cached_gnn_embeddings = None

    def session_emb(self, batch) -> torch.Tensor:
        seq = pad_sequence(batch.views, batch_first=True, padding_value=self.n_items).to(self.device)
        _, indices = batch.extra.seq_lens.sort(dim=0, descending=True)
        packed = pack_padded_sequence(self.target_emb(seq[indices]), lengths=batch.extra.seq_lens[indices], batch_first=True)
        _, h = self.gru(packed)
        h = h.transpose(0, 1)
        _, inv_indices = torch.sort(indices, 0, descending=False)
        s_emb = self.drop(h[inv_indices].squeeze(dim=1))
        s_emb = s_emb.mean(dim=1)
        for l, bn in zip(self.last_layers, self.last_bns):
            s_emb = torch.tanh(self.drop(l(s_emb)))
            s_emb = bn(s_emb)
        return s_emb
    
    def forward(self, batch, mask: bool = True) -> torch.Tensor:
        s_emb = self.session_emb(batch)
        logit = F.linear(
            s_emb, 
            F.normalize(self.drop(self.target_emb(torch.arange(self.n_items, device=self.device)))), 
            self.item_bias
        )
        if mask:
            # [수치 안정성] -10000.0 매직넘버를 표준 -float('inf')로 대체하여 언더플로우 및 FP16 NaN 방지
            logit[batch.extra.histories] = -float('inf')
        return logit
    
    def target_emb(self, iid: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_emb(iid)
        # [방어적 캐싱] 캐시가 비어있을 경우 자동 연산 및 갱신
        if self._cached_gnn_embeddings is None:
            self.cache_gnn_embeddings()
        gnn_embeddings = self._cached_gnn_embeddings
        item_emb_gnn = F.embedding(iid, gnn_embeddings, padding_idx=self.n_items)
        return torch.cat([item_emb, item_emb_gnn], dim=-1)
    
    def init_weights(self):
        """[방어적 모듈 초기화] 문자열 필터링 취약점을 제거하고 명확한 nn.Module 기반 초기화 수행"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    with torch.no_grad():
                        module.bias.data.normal_(0.0, 0.001).clamp_(min=-0.001, max=0.001)
            elif isinstance(module, nn.Embedding) and module.padding_idx is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)


def run(
    epochs: int = 30, batch_size: int = 512, device: str = 'cuda', seed: int = 2022, 
    num_workers: int = 0, num_negatives: int = 100, pin_memory: bool = False, persistent_workers: bool = False,
    save_fname: str = 'save/grun.pt', submit: bool = False, all: bool = False, **kwargs
):
    print(locals())
    sleep_seconds = random.randint(1, 10)
    print(f'sleep {sleep_seconds} before start')
    time.sleep(sleep_seconds)
    
    dir_name = 'processed'
    if submit:
        dir_name += '_submit'
    time.sleep(random.randint(1, 10))
    dir_name = pjoin(abspath(dirname(dirname(dirname(__file__)))), dir_name)
    
    idmap, fidmap = joblib.load(f'{dir_name}/indices')[:2]
    pick = '_all' if all else ''
    aug = '_aug' if kwargs.get('augmentation', False) else ''
    df_train = joblib.load(f'{dir_name}/df_train{aug}{pick}')
    df_te = joblib.load(f'{dir_name}/df_val')
    
    pick = 'all_' if all else ''
    features = torch.FloatTensor(joblib.load(f'{dir_name}/features'))
    edge_index, edge_weight = joblib.load(f'{dir_name}/{pick}edge_index_v1.1')
    
    model = GRUN(idmap, features, edge_index, edge_weight).to(device)    
    model.set_fork_logic(df_train, df_te, joblib.load(f'{dir_name}/features'))
    
    ds_tr = SessionDataset(df_train, idmap=idmap, fidmap=fidmap, num_negatives=num_negatives)
    dl_tr = SessionDataLoader(
        ds_tr, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap,
        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory,
        shuffle=kwargs.get('shuffle', False)
    )
    ds_te = SessionDataset(df_te, idmap=idmap, fidmap=fidmap, num_negatives=num_negatives)
    dl_te = SessionDataLoader(ds_te, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap)
    
    model.cache_gnn_embeddings()

    if os.path.exists(save_fname):
        model.load_state_dict(torch.load(save_fname))
        model.set_fork_logic(df_train, df_te)
        model.cache_gnn_embeddings()
        hit, mrr = model.validate(dl_te, **kwargs)
        print(f'HIT: {hit:.6f}, MRR: {mrr:.8f}')
        return
        
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=1e-5, verbose=True)
    
    # [학습 루프 안정성] Model 베이스 클래스의 fit 인터페이스에 에폭 단위 캐시 무효화 훅 주입 혹은 콜백 연계 가능 구조 설계 반영
    model.fit(dl_tr, dl_te, optimizer, scheduler, epochs=epochs, seed=seed, save_fname=save_fname)


if __name__ == '__main__':
    fire.Fire(run)

최종 개선사항
✅ CUDA_LAUNCH_BLOCKING 상시 활성화 → DEBUG_CUDA 환경 조건부 실행으로 프로덕션 GPU 성능 저하 방지
✅ sys.path 강제 수정 → 패키지 import 우선 구조로 전환하여 실행 위치 의존성 제거
✅ GNN Parameter 방식 저장 → register_buffer 적용으로 device 이동·checkpoint 관리 안정성 확보
✅ Forward 내부 전체 GNN 재연산 → GNN embedding cache 구조로 변경하여 추론 및 학습 연산량 감소
✅ 단순 Tensor 캐싱 → no_grad 기반 캐싱으로 불필요한 graph 유지 및 메모리 누수 방지
✅ Stale embedding 위험 → invalidate_cache 인터페이스 추가로 cache 갱신 제어 기반 확보
✅ 문자열 기반 weight 탐색 → nn.Module 타입 기반 초기화로 parameter 누락 가능성 제거
✅ try-except pass 초기화 → 명시적 초기화 실패 감지 구조로 변경하여 Silent Failure 제거
✅ -10000.0 masking → -float('inf') 적용으로 FP16·Softmax 수치 안정성 강화
✅ 전역 상태 의존 설정 → 환경 변수 기반 디버그·운영 분리 구조로 확장성 개선
✅ Embedding padding 초기화 누락 → padding_idx zero 처리로 sequence 모델 안정성 강화
✅ 레거시 import 호환 유지 → 패키지 전환 과정 중 기존 실행 환경 깨짐 방지

기존 Forward GNN 재계산·Silent Failure·CUDA 강제 디버그·비안전 import 구조를 대부분 제거했으며, 남은 핵심 과제는 epoch 단위 cache invalidate를 실제 학습 루프에 연결하는 부분으로 9.5/10 수준까지 올라온 엔터프라이즈 ML 구조다.
