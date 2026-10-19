from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class SEBlock(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.fc2 = nn.Conv1d(bottleneck, channels, kernel_size=1)

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class Res2Block(nn.Module):
    def __init__(self, channels, scale=8, dilation=2):
        super().__init__()
        self.scale = scale
        self.width = channels // scale
        self.convs = nn.ModuleList(
            [nn.Conv1d(self.width, self.width, kernel_size=3, padding=dilation, dilation=dilation) for _ in range(scale - 1)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])

    def forward(self, x):
        xs = torch.split(x, self.width, dim=1)
        y = []
        y.append(xs[0])
        out = xs[0]
        for i in range(self.scale - 1):
            if i == 0:
                out = xs[i + 1]
            else:
                out = xs[i + 1] + out
            out = F.relu(self.bns[i](self.convs[i](out)))
            y.append(out)
        return torch.cat(y, dim=1)


class SERes2Block(nn.Module):
    def __init__(self, channels, scale=8, dilation=2, se_bottleneck=128):
        super().__init__()
        self.in_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.in_bn = nn.BatchNorm1d(channels)
        self.res2 = Res2Block(channels, scale=scale, dilation=dilation)
        self.out_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.out_bn = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels, bottleneck=se_bottleneck)

    def forward(self, x):
        identity = x
        out = F.relu(self.in_bn(self.in_conv(x)))
        out = self.res2(out)
        out = F.relu(self.out_bn(self.out_conv(out)))
        out = self.se(out)
        return out + identity


class AttentiveStatsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.tdnn = nn.Conv1d(channels, attention_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.att = nn.Conv1d(attention_channels, channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C, T]
        a = self.att(self.tanh(self.tdnn(x)))
        a = torch.softmax(a, dim=2)
        mean = torch.sum(a * x, dim=2)
        var = torch.sum(a * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPATDNN(nn.Module):
    def __init__(self, n_mels=128, channels=512, emb_dim=192, num_classes=10):
        super().__init__()
        self.n_mels = n_mels
        self.frame1 = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.block1 = SERes2Block(channels, scale=8, dilation=2)
        self.block2 = SERes2Block(channels, scale=8, dilation=3)
        self.block3 = SERes2Block(channels, scale=8, dilation=4)
        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.pool = AttentiveStatsPooling(channels)
        self.bn_pool = nn.BatchNorm1d(channels * 2)
        self.emb = nn.Linear(channels * 2, emb_dim)
        self.bn_emb = nn.BatchNorm1d(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward_features(self, x):
        # x: [B, n_mels, T]
        x = self.frame1(x)
        x1 = self.block1(x)
        x2 = self.block2(x + x1)
        x3 = self.block3(x + x1 + x2)
        x = self.mfa(torch.cat([x1, x2, x3], dim=1))
        x = self.pool(x)
        x = self.bn_pool(x)
        x = self.emb(x)
        x = self.bn_emb(x)
        return x

    def forward(self, x):
        e = self.forward_features(x)
        return self.classifier(e)


class AAMSoftmaxHead(nn.Module):
    """
    Additive Angular Margin Softmax head.
    - Input embedding should be shape [B, D]
    - Returns scaled logits for CrossEntropy
    """
    def __init__(self, emb_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(self.num_classes, self.emb_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor = None, apply_margin: bool = True):
        # cosine between normalized embedding and normalized class weights
        emb_n = F.normalize(emb, p=2, dim=1)
        w_n = F.normalize(self.weight, p=2, dim=1)
        cosine = torch.matmul(emb_n, w_n.transpose(0, 1))
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        if apply_margin and labels is not None:
            theta = torch.acos(cosine)
            target_cos = torch.cos(theta + self.margin)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            logits = one_hot * target_cos + (1.0 - one_hot) * cosine
        else:
            logits = cosine
        return logits * self.scale


class ECAPAFeatureDataset(Dataset):
    def __init__(self, items, feature_root: Path = None, shard_root: Path = None, input_layout='BMT'):
        self.items = items
        self.feature_root = feature_root
        self.shard_root = shard_root
        self.input_layout = input_layout  # BMT: [B, Mel, Time], BTM: [B, Time, Mel]

        if feature_root is None and shard_root is None:
            raise ValueError('either feature_root or shard_root must be provided')

        self.mapping = None
        self._cache_name = None
        self._cache_arr = None
        if shard_root is not None:
            idx_csv = shard_root / 'feature_shard_index.csv'
            if not idx_csv.exists():
                raise FileNotFoundError(f'Missing shard index: {idx_csv}')
            mapping = {}
            with idx_csv.open('r', encoding='utf-8-sig', newline='') as f:
                for r in csv.DictReader(f):
                    mapping[r['wav_path']] = (r['shard_name'], int(r['index_in_shard']))
            self.mapping = mapping

    def __len__(self):
        return len(self.items)

    def _load_feat(self, wav_path: str):
        if self.shard_root is not None:
            if wav_path not in self.mapping:
                raise KeyError(f'wav_path not found in shard index: {wav_path}')
            shard_name, idx = self.mapping[wav_path]
            if self._cache_name != shard_name:
                self._cache_arr = np.load(self.shard_root / shard_name, mmap_mode='r')
                self._cache_name = shard_name
            arr = np.array(self._cache_arr[idx], copy=True)
        else:
            rel = Path(wav_path).parent.name + '/' + Path(wav_path).stem + '.pt'
            fp = self.feature_root / rel
            if not fp.exists():
                raise FileNotFoundError(f'feature file not found: {fp}')
            arr_t = torch.load(fp, map_location='cpu')
            arr = arr_t.numpy() if isinstance(arr_t, torch.Tensor) else np.array(arr_t)
            arr = np.array(arr, copy=True)
        return arr

    def _to_mel_time(self, arr: np.ndarray):
        # expected possible shapes: [M,T], [1,M,T], [3,M,T]
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3):
                arr = arr[0]  # remove channel
            else:
                raise ValueError(f'unsupported 3D feature shape: {arr.shape}; expected [1,M,T] or [3,M,T]')
        elif arr.ndim != 2:
            raise ValueError(f'unsupported feature ndim={arr.ndim}, shape={arr.shape}')

        # arr now [M,T]
        x = torch.from_numpy(arr).float()
        if self.input_layout == 'BTM':
            x = x.transpose(0, 1)  # [T,M]
        return x

    def __getitem__(self, idx):
        it = self.items[idx]
        wav_path = str(it.wav_path)
        arr = self._load_feat(wav_path)
        x = self._to_mel_time(arr)
        y = torch.tensor(it.label, dtype=torch.long)
        return x, y, wav_path, it.class_name
