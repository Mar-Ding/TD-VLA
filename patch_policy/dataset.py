"""Pusht data loading + window sampling over precomputed WeMM embeddings."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import List, Tuple

import einops
import torch
from torch.utils.data import Dataset


# ---- raw data file readers ------------------------------------------------


def load_seq_lengths(data_directory: str | Path) -> List[int]:
    with open(Path(data_directory) / "seq_lengths.pkl", "rb") as f:
        return list(pickle.load(f))


def load_actions(
    data_directory: str | Path, n_episodes: int, relative: bool = False
) -> List[torch.Tensor]:
    """Per-episode action tensor [L, act_dim] (zero-padding beyond seq length removed)."""
    name = "rel_actions.pth" if relative else "abs_actions.pth"
    acts = torch.load(Path(data_directory) / name, weights_only=True)
    seq_lengths = load_seq_lengths(data_directory)
    return [acts[i, : seq_lengths[i]].float() for i in range(n_episodes)]


def load_episode_frames(data_directory: str | Path, idx: int) -> torch.Tensor:
    """Raw frames [L, 3, H, W] float 0..1 (WeMM's own preprocessing handles resize/norm)."""
    obs = torch.load(Path(data_directory) / "obses" / f"episode_{idx:03d}.pth")
    obs = einops.rearrange(obs, "T H W C -> T C H W").float() / 255.0
    return obs


def frame_count(data_directory: str | Path) -> int:
    return sum(load_seq_lengths(data_directory))


# ---- precomputed embedding cache ------------------------------------------


def save_embedding_cache(path: str | Path, meta: dict, episodes: List[torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "episodes": episodes}, path)


def load_embedding_cache(
    path: str | Path, check_meta: dict | None = None
) -> Tuple[dict, List[torch.Tensor]]:
    """Returns (meta, episodes). Warns on meta mismatch."""
    data = torch.load(path, weights_only=False)
    meta, episodes = data["meta"], data["episodes"]
    if check_meta is not None:
        for k, v in check_meta.items():
            if meta.get(k) != v:
                print(
                    f"[warn] cache meta mismatch on '{k}': cache={meta.get(k)} expected={v}; "
                    "delete the cache file if the dataset/model changed."
                )
    return meta, episodes


# ---- WeMM-DINO Fusion: in-memory DINO-P256 precompute ----------------------
def precompute_dino_in_memory(
    dino_encoder,
    data_directory: str | Path,
    n_episodes: int,
    device,
    encode_batch_size: int = 32,
    verbose: bool = True,
) -> List[torch.Tensor]:
    """DINO-P256 patch tokens for all episodes, in RAM (no disk cache).

    Disk is 100% full (~5.6G free; a fp32 P256 cache would be ~10GB), so unlike
    WeMM (small P=1 cache) the DINO-P256 tokens are precomputed in-memory once at
    data load — the same in-memory path the base DINO-P256 baseline used (base
    never wrote a DINO cache either). Per-episode [L, 256, 384] fp32, computed
    from the SAME obses/episode_XXX.pth frames as the WeMM cache -> per-frame
    aligned with wemm_cache by construction. Deterministic (DINO frozen, fp32).
    """
    episodes = []
    t0 = time.time()
    for ei in range(n_episodes):
        frames = load_episode_frames(data_directory, ei)          # [L,3,224,224] fp32 0..1
        L = frames.shape[0]
        emb = torch.zeros(L, dino_encoder.n_patches, dino_encoder.output_dim)
        for s in range(0, L, encode_batch_size):
            chunk = frames[s : s + encode_batch_size].to(device)
            emb[s : s + encode_batch_size] = dino_encoder(chunk).cpu()  # [bs,256,384]
        assert torch.isfinite(emb).all()
        episodes.append(emb)
        if verbose and (ei + 1) % 20 == 0:
            print(f"  [dino-mem] {ei+1}/{n_episodes} episodes "
                  f"({sum(e.shape[0] for e in episodes)} frames, {time.time()-t0:.0f}s)")
    if verbose:
        print(f"[dino-mem] {n_episodes} episodes, {sum(e.shape[0] for e in episodes)} frames, "
              f"{time.time()-t0:.0f}s")
    return episodes


# ---- WeMM-Repeat-P256 (Global-to-Patch Token Expansion) -------------------
def expand_repeat_patches(obs: torch.Tensor, target_patches: int) -> torch.Tensor:
    """Expand the single global token to target_patches copies BEFORE the policy projection.

    [..., 1, D] -> [..., target_patches, D] with IDENTICAL feature content (pure
    copy, no learned transform). Only the policy-facing P changes: the encoder /
    cache stay P=1 (no new cache, no 256× disk copy). Tokens become distinguishable
    solely through the Patch Policy's existing wpe positional embedding
    (token_i = W(z) + wpe(t*P + i)). Materializes a new tensor per batch.
    """
    p = obs.shape[-2]
    assert p == 1, f"repeat expansion expects P=1 input, got P={p}"
    assert target_patches % p == 0, f"target_patches {target_patches} not multiple of P={p}"
    return obs.repeat_interleave(target_patches // p, dim=-2)


# ---- windowed training/val dataset ----------------------------------------

# base utils/inference.py 语义：前向/后向补帧（VQ-BeT 受控对照需逐样本对齐）
def repeat_start_to_length(x: torch.Tensor, length: int, dim: int = 0) -> torch.Tensor:
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    first = x.index_select(dim, torch.tensor(0, device=x.device))
    shape = [1] * len(x.shape)
    shape[dim] = pad_size
    return torch.cat([first.repeat(*shape), x], dim=dim)


def repeat_end_to_length(x: torch.Tensor, length: int, dim: int = 0) -> torch.Tensor:
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    last = x.index_select(dim, torch.tensor(x.shape[dim] - 1, device=x.device))
    shape = [1] * len(x.shape)
    shape[dim] = pad_size
    return torch.cat([x, last.repeat(*shape)], dim=dim)


class WindowDataset(Dataset):
    """Slides windows over precomputed per-episode embeddings.

    Replicates base `TrajectorySlicerDataset` with pad_seq_length=True and
    vqbet_get_future_action_chunk=False:
      * full windows [s, s+T) for s in 0..L-T        (obs always full length)
      * 4 start-virtual windows [0, 1..T-1]          (obs repeat-start padded)
      * action chunk [s, s+T+W-1), repeat-end padded to T+W-1 at episode tail
    -> L windows per episode, identical to the baseline data pipeline.
    """

    def __init__(
        self,
        episodes_emb: List[torch.Tensor],
        actions: List[torch.Tensor],
        obs_window_size: int,
        action_window_size: int,
        episode_indices: List[int] | None = None,
    ):
        self.episodes_emb = episodes_emb
        self.actions = actions
        self.T = obs_window_size
        self.W = action_window_size
        self.need = obs_window_size + action_window_size - 1  # T+W-1

        indices = episode_indices if episode_indices is not None else list(range(len(episodes_emb)))
        self.samples: List[Tuple[int, int, int]] = []  # (ei, start, end)
        for ei in indices:
            L = episodes_emb[ei].shape[0]
            for end in range(self.T - 1):  # start-virtual windows [0, end+1)
                self.samples.append((ei, 0, end + 1))
            if L - self.T >= 0:  # full windows [start, start+T)
                for start in range(L - self.T + 1):
                    self.samples.append((ei, start, start + self.T))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ei, start, end = self.samples[i]
        obs = self.episodes_emb[ei][start:end]  # <= T frames
        act = self.actions[ei][start : end - 1 + self.W]  # <= T+W-1 frames
        if end - start < self.T:
            # start-virtual window: base core.py L176-184 —— obs 与 act 均前向补（repeat 首帧/首动作）
            obs = repeat_start_to_length(obs, self.T, dim=0)  # [T, P, D]
            act = repeat_start_to_length(act, self.need, dim=0)  # [T+W-1, A]
        else:
            # full window：仅当 act 越过后部时后向补（repeat 末动作），base core.py L203-210
            act = repeat_end_to_length(act, self.need, dim=0)  # [T+W-1, A]
        return obs, act

    def _frame_indices(self, k: int) -> Tuple[List[int], List[int]]:
        """Exact per-window (obs, act) frame-index sequences __getitem__ yields.

        Single source of truth mirroring __getitem__'s slicing+padding, for the
        dataset-parity oracle to compare against base core.py semantics.
        """
        ei, start, end = self.samples[k]
        L = self.episodes_emb[ei].shape[0]
        obs_idx = list(range(start, end))
        if end - start < self.T:
            obs_idx = [0] * (self.T - len(obs_idx)) + obs_idx
        act_idx = list(range(start, min(end - 1 + self.W, L)))
        if end - start < self.T:
            act_idx = [act_idx[0]] * (self.need - len(act_idx)) + act_idx
        elif len(act_idx) < self.need:
            act_idx = act_idx + [act_idx[-1]] * (self.need - len(act_idx))
        return obs_idx, act_idx


class FusionWindowDataset(WindowDataset):
    """DINO dense tokens + WeMM global cond, per-frame aligned (spec §1).

    Uses the SAME (ei, start, end) window samples as WindowDataset — the two
    streams are aligned BY CONSTRUCTION (same episode, same frame slice, same
    padding). DINO tokens [L,256,384] come from precompute_dino_in_memory (no
    disk cache); WeMM cond [L,1,2048] from the existing disk cache. Returns
    (dino_obs, wemm_cond, act) with bit-identical obs/act slicing vs baseline.
    """

    def __init__(
        self,
        dino_episodes: List[torch.Tensor],
        cond_episodes: List[torch.Tensor],
        actions: List[torch.Tensor],
        obs_window_size: int,
        action_window_size: int,
        episode_indices: List[int] | None = None,
    ):
        super().__init__(
            dino_episodes, actions, obs_window_size, action_window_size, episode_indices
        )
        self.cond_episodes = cond_episodes
        # strict per-frame alignment guard (DINO length == WeMM cond length per episode)
        seen = set()
        for ei, _, _ in self.samples:
            if ei in seen:
                continue
            seen.add(ei)
            assert dino_episodes[ei].shape[0] == cond_episodes[ei].shape[0], (
                f"ep{ei}: DINO frames {dino_episodes[ei].shape[0]} != WeMM cond "
                f"{cond_episodes[ei].shape[0]} (per-frame alignment violated)"
            )

    def __getitem__(self, i):
        obs, act = super().__getitem__(i)  # WindowDataset: dino obs [T,256,384] + act (same padding)
        ei, start, end = self.samples[i]
        cond = self.cond_episodes[ei][start:end]  # [<=T,1,2048]
        if end - start < self.T:
            cond = repeat_start_to_length(cond, self.T, dim=0)  # same repeat-start padding
        return obs, cond, act


def split_episodes(
    n_episodes: int, train_fraction: float, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Seeded random permutation split (mirrors base random_split_traj with seed 42).

    Returns (train_indices, val_indices).
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_episodes, generator=g).tolist()
    n_train = int(n_episodes * train_fraction)
    return idx[:n_train], idx[n_train:]
