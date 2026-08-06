import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k == None:
            self.cache_k = new_k # (batch, seq_len, model_dim)
            self.cache_v = new_v # (batch, seq_len, model_dim)
        else: 
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1) # (batch, seq_len+1, model_dim)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1) # (batch, seq_len+1, model_dim)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Return (rounded output, kv_cache)
        
        # x -> (B, T, A), where B=batch, T=seq_len, and A=model_dim
        q = self.q_proj(x) # -> (B, T, A)
        k = self.k_proj(x) # -> (B, T, A)
        v = self.v_proj(x) # -> (B, T, A)
        if kv_cache == None:
            kv_cache = KVCache()
        full_k, full_v = kv_cache.update(k, v)
        scores = q @ torch.transpose(full_k, -2, -1) / (q.shape[2]**0.5) # (B,T=1,T=N)
        scores = torch.softmax(scores, dim=-1)
        return (torch.round(scores @ full_v, decimals=4), kv_cache) # => (B, T=1, A)


