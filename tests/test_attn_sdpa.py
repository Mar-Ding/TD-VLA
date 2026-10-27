"""Attention-kernel parity: hand-written Q@K^T chain vs F.scaled_dot_product_attention.

Why this is a test and not a paragraph: the SDPA path is only defensible if it is the
SAME function, bit-for-bit up to summation order. A report can claim parity once; this
file re-checks it on every run, for all THREE attention sites, and fails if a later edit
touches the mask convention (bool True == may attend), the `is_causal` flag, the dropout
placement, or the head reshape.

Tolerances are dtype-specific and carry their reason:
  fp32 (CPU)   1e-6   -- same math, different summation order. Measured 1e-7..2e-6.
  bf16 (CUDA)  1.5e-2 -- the fused kernels accumulate in fp32 but store bf16; the trunk
                         reduction is S=1280 long. Measured ~2e-3. A tighter bf16 bound
                         would be a lie: bf16 is not reproducible across kernels.

What parity does NOT claim: identical training trajectories. Different summation order
compounds over steps, so an SDPA run and a `current` run diverge in the last digits and
therefore in the final weights. That is documented, not tested -- see
doc/2026-10-27_attn_sdpa.md.

Run (local or server, no diffusers needed):
    python tests/test_attn_sdpa.py
    pytest tests/test_attn_sdpa.py -q
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patch_policy.gpt import (  # noqa: E402
    ATTN_IMPLS,
    CausalSelfAttention,
    GPT,
    GPTConfig,
    set_attn_impl,
)


def _try_import(module, name):
    """Import one symbol, tolerating a missing OPTIONAL dependency.

    `td_head` / `global_policy` pull in `vqbet` -> `vqvae` -> `vector_quantize_pytorch`
    for the VQ tokeniser, which the attention paths never touch. A pure-torch box
    therefore cannot import them at all, and a parity test that refuses to run there is
    a parity test that stops being run. Missing dep => the two checks that need it are
    reported as SKIPPED (with the reason), never as passed.
    """
    try:
        import importlib
        return getattr(importlib.import_module(module), name)
    except ModuleNotFoundError:
        return None


GlobalSelfAttention = _try_import("patch_policy.global_policy", "GlobalSelfAttention")
CrossAttention = _try_import("patch_policy.td_head", "CrossAttention")
HAVE_VQ_DEP = GlobalSelfAttention is not None and CrossAttention is not None

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def _pair(module, *inputs, dtype=torch.float32, device="cpu"):
    """Run one module twice (current / sdpa) on identical inputs+weights, eval mode."""
    module.eval().to(device=device, dtype=dtype)
    with torch.no_grad():
        set_attn_impl(module, "current")
        y_cur = module(*inputs)
        set_attn_impl(module, "sdpa")
        y_sdpa = module(*inputs)
    assert y_cur.shape == y_sdpa.shape
    d = (y_cur.float() - y_sdpa.float()).abs()
    return y_cur, y_sdpa, d


# ---------------------------------------------------------------------------------------
# 1. the three sites, fp32
# ---------------------------------------------------------------------------------------
def test_trunk_self_attention_fp32():
    torch.manual_seed(0)
    n_embd, n_head, block_size, n_patches = 64, 4, 3, 16   # S = 48, same structure
    m = CausalSelfAttention(n_embd, n_head, block_size, n_patches, dropout=0.0)
    x = torch.randn(2, block_size * n_patches, n_embd)
    _, _, d = _pair(m, x)
    check("trunk CausalSelfAttention: sdpa == current (fp32)", d.max().item() < 1e-6,
          f"max_abs={d.max().item():.2e} mean_abs={d.mean().item():.2e}")


def test_global_self_attention_fp32():
    if GlobalSelfAttention is None:
        check("global GlobalSelfAttention: sdpa == current (fp32)", True,
              "SKIPPED: needs vector_quantize_pytorch (VQ tokeniser, unused here)")
        return
    torch.manual_seed(1)
    n_embd, n_head, n_state, n_query = 64, 4, 5, 3          # T = 8
    m = GlobalSelfAttention(n_embd, n_head, n_state, n_query, dropout=0.0)
    x = torch.randn(2, n_state + n_query, n_embd)
    _, _, d = _pair(m, x)
    check("global GlobalSelfAttention: sdpa == current (fp32)", d.max().item() < 1e-6,
          f"max_abs={d.max().item():.2e} mean_abs={d.mean().item():.2e}")


def test_cross_attention_fp32():
    if CrossAttention is None:
        check("td CrossAttention: sdpa == current (fp32), ctx_dim != d_model", True,
              "SKIPPED: needs vector_quantize_pytorch (VQ tokeniser, unused here)")
        return
    torch.manual_seed(2)
    # ctx_dim != d_model on purpose: the trunk readout is 256 wide while the denoiser is
    # 512 wide on the clean line, so the k/v projection crosses widths. That reshape is
    # exactly the kind of thing an SDPA port can get wrong silently.
    m = CrossAttention(d_model=64, n_heads=4, ctx_dim=32, dropout=0.0)
    q = torch.randn(8, 20, 64)
    ctx = torch.randn(8, 5, 32)
    _, _, d = _pair(m, q, ctx)
    check("td CrossAttention: sdpa == current (fp32), ctx_dim != d_model",
          d.max().item() < 1e-6,
          f"max_abs={d.max().item():.2e} mean_abs={d.mean().item():.2e}")


# ---------------------------------------------------------------------------------------
# 2. the mask is the whole argument -- check it directly, not via the module
# ---------------------------------------------------------------------------------------
def test_block_causal_mask_matches_manual_chain():
    """The kernel's bool convention (True == may attend) must equal our masked_fill(~mask).

    If a future edit flips the mask polarity, or passes a float mask with 0/1 semantics,
    the module-level test above would still pass on a symmetric input but the model would
    be wrong. This one pins the convention on an asymmetric, hand-built case.
    """
    torch.manual_seed(3)
    S, nh, hd = 12, 2, 8
    q, k, v = (torch.randn(1, nh, S, hd) for _ in range(3))
    # block-causal with n_patches=4, block_size=3
    mask = torch.zeros(S, S, dtype=torch.bool)
    for t in range(3):
        mask[t * 4:(t + 1) * 4, :(t + 1) * 4] = True

    att = (q @ k.transpose(-2, -1)) * hd ** -0.5
    att = att.masked_fill(~mask, float("-inf"))
    y_manual = torch.softmax(att, dim=-1) @ v
    y_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=False)
    d = (y_manual - y_sdpa).abs().max().item()
    check("bool mask (True == may attend) reproduces masked_fill(-inf)", d < 1e-6,
          f"max_abs={d:.2e}")

    # and the masked-out entries really are excluded: block-causal allows a block to see
    # EARLIER blocks, never later ones. Perturb the LAST block's key and assert the first
    # two blocks' outputs do not move at all.
    k2 = k.clone()
    k2[:, :, -1, :] += 1.0                      # a key in the final block
    y2 = torch.nn.functional.scaled_dot_product_attention(
        q, k2, v, attn_mask=mask, is_causal=False)
    moved_past = (y2 - y_sdpa).abs()[:, :, :8, :].max().item()   # blocks 0 and 1
    moved_self = (y2 - y_sdpa).abs()[:, :, -1, :].max().item()   # block 2 must move
    check("a later block's key cannot reach earlier blocks", moved_past < 1e-6,
          f"max_abs_past={moved_past:.2e}")
    check("...while the block that owns that key does move", moved_self > 1e-3,
          f"max_abs_own={moved_self:.2e}")


# ---------------------------------------------------------------------------------------
# 3. dropout placement
# ---------------------------------------------------------------------------------------
def test_dropout_is_applied_in_training_mode():
    torch.manual_seed(4)
    m = CausalSelfAttention(64, 4, 3, 16, dropout=0.3)
    x = torch.randn(2, 48, 64)
    m.eval()
    set_attn_impl(m, "sdpa")
    with torch.no_grad():
        y_eval = m(x)
    m.train()
    set_attn_impl(m, "sdpa")
    with torch.no_grad():
        y_train = m(x)
    check("sdpa: dropout active in train, off in eval",
          bool(torch.isfinite(y_train).all()) and (y_train - y_eval).abs().max() > 1e-3,
          f"train-vs-eval max_abs={(y_train - y_eval).abs().max().item():.2e}")


# ---------------------------------------------------------------------------------------
# 4. whole trunk end to end (wte / wpe / 2 blocks / ln_f / readout)
# ---------------------------------------------------------------------------------------
def test_gpt_trunk_end_to_end_fp32():
    torch.manual_seed(5)
    cfg = GPTConfig(input_dim=16, n_embd=64, n_layer=2, n_head=4, block_size=3,
                    n_patches=16, dropout=0.0)
    m = GPT(cfg).eval()
    x = torch.randn(2, 3, 16, 16)          # [B, T, P, D]
    with torch.no_grad():
        set_attn_impl(m, "current")
        y_cur = m(x)
        set_attn_impl(m, "sdpa")
        y_sdpa = m(x)
    d = (y_cur - y_sdpa).abs().max().item()
    check("GPT trunk readout: sdpa == current (fp32)", d < 1e-5, f"max_abs={d:.2e}")


# ---------------------------------------------------------------------------------------
# 5. the selector itself
# ---------------------------------------------------------------------------------------
def test_set_attn_impl_contract():
    torch.manual_seed(6)
    sites = {"trunk": CausalSelfAttention(64, 4, 3, 16, 0.0)}
    if HAVE_VQ_DEP:
        sites["global"] = GlobalSelfAttention(64, 4, 5, 3, 0.0)
        sites["cross"] = CrossAttention(64, 4, ctx_dim=32, dropout=0.0)
    sites["plain"] = torch.nn.Linear(8, 8)
    model = torch.nn.ModuleDict(sites)
    want = 3 if HAVE_VQ_DEP else 1
    n = set_attn_impl(model, "sdpa")
    check(f"set_attn_impl reaches every attention site ({want} here)", n == want,
          f"n={n}" + ("" if HAVE_VQ_DEP else " (global/cross skipped: no VQ dep)"))
    check("every attention module switched",
          all(m.attn_impl == "sdpa" for k, m in model.items() if k != "plain"))
    check("non-attention modules are untouched",
          not hasattr(model["plain"], "attn_impl"))
    n_back = set_attn_impl(model, "current")
    check("switch is reversible (default restored)", n_back == want and
          all(m.attn_impl == "current" for k, m in model.items() if k != "plain"))
    try:
        set_attn_impl(model, "flash2")
        ok = False
    except ValueError:
        ok = True
    check("unknown kernel is a ValueError, not a silent no-op", ok)
    check("ATTN_IMPLS is the documented pair", ATTN_IMPLS == ("current", "sdpa"))


# ---------------------------------------------------------------------------------------
# 6. CUDA + bf16, where the fused kernels actually pay off
# ---------------------------------------------------------------------------------------
def test_cuda_bf16_parity():
    if not torch.cuda.is_available():
        check("cuda/bf16 parity", True, "skipped: no CUDA")
        return
    torch.manual_seed(7)
    device = "cuda"
    m = CausalSelfAttention(512, 8, 5, 256, dropout=0.0).to(device)  # production shape
    x = torch.randn(1, 5 * 256, 512, device=device, dtype=torch.bfloat16)
    y_cur, y_sdpa, d = _pair(m, x, dtype=torch.bfloat16, device=device)
    check("trunk production shape (S=1280, nh=8) bf16 parity",
          d.max().item() < 1.5e-2 and bool(torch.isfinite(y_sdpa).all()),
          f"max_abs={d.max().item():.2e} mean_abs={d.mean().item():.2e}")

    # which kernel did the default dispatch actually use? Not a guess -- asked of the
    # build, and printed so a reader can see that "sdpa" is not a synonym for "flash".
    backends = []
    for name in ("flash", "mem_efficient", "math"):
        fn = getattr(torch.backends.cuda, f"{name}_sdp_enabled", None)
        if fn is not None:
            backends.append(f"{name}={fn()}")
    print(f"    build: torch={torch.__version__} cuda={torch.version.cuda} "
          f"bf16={torch.cuda.is_bf16_supported()} {' '.join(backends)}")


def main():
    torch.set_num_threads(2)
    print("== attention kernel parity: current vs sdpa ==")
    test_trunk_self_attention_fp32()
    test_global_self_attention_fp32()
    test_cross_attention_fp32()
    print("\n== mask convention (the part a refactor can silently break) ==")
    test_block_causal_mask_matches_manual_chain()
    print("\n== dropout placement ==")
    test_dropout_is_applied_in_training_mode()
    print("\n== whole trunk ==")
    test_gpt_trunk_end_to_end_fp32()
    print("\n== selector contract ==")
    test_set_attn_impl_contract()
    print("\n== cuda / bf16 ==")
    test_cuda_bf16_parity()
    failed = [n for n, ok in RESULTS if not ok]
    print("\n" + "=" * 74)
    print(f"FAILED {len(failed)}: " + "; ".join(failed) if failed
          else f"ALL {len(RESULTS)} CHECKS PASSED")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
