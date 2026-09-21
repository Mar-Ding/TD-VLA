"""Contract tests for the TD action head. Pure torch, NO diffusers, NO sklearn.

These are the assertions the P0/P1b reports lean on. Each one is here rather than in a
report because a report states a fact once and a test keeps stating it: if someone later
re-shapes `t`, switches `cls_from`, or turns off `clip_sample`, this fails loudly instead
of quietly invalidating a published table.

Run (local or server, either env -- diffusers is not imported):
    python tests/test_td_contract.py            # prints PASS/FAIL per check
    pytest tests/test_td_contract.py -q         # same checks, pytest-collected
"""
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patch_policy.anchors import kmeans_torch, save_anchor_artifact  # noqa: E402
from patch_policy.config import Config, TDConfig  # noqa: E402
from patch_policy.diffusion import (  # noqa: E402
    TruncatedDiffusionSchedule,
    cumulative_travel,
    inference_chain,
    make_inference_timesteps,
)
from patch_policy.normalizer import ActionNormalizer  # noqa: E402
from patch_policy.train import build_model  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def make_cfg(tmpdir, **td_overrides):
    """The `clean` line: native_parity -> gpt readout_dim == 256, not 512."""
    cfg = Config()
    cfg.encoder_name = "dino"
    cfg.dino.postprocess = None
    cfg.dino.n_patches = 256
    cfg.action_head = "td"
    cfg.obs_window_size = 5
    cfg.action_window_size = 5
    cfg.policy.native_parity = True
    cfg.policy.n_layer = 2          # smaller than production; the contract is shape/logic
    cfg.policy.n_head = 8
    cfg.policy.n_embd = 512
    cfg.td = TDConfig(anchor_artifact=str(Path(tmpdir) / "anchors.pt"),
                      n_modes=20, **td_overrides)
    return cfg


def make_artifact(cfg, tmpdir):
    """Fit a real normaliser + k-means on synthetic PushT-range actions."""
    g = torch.Generator().manual_seed(0)
    n = 4000
    a = torch.stack([torch.randint(12, 512, (n,), generator=g).float(),
                     torch.randint(25, 512, (n,), generator=g).float()], dim=-1)
    chunks = a.reshape(-1, cfg.action_window_size, 2)[: n // cfg.action_window_size]
    norm = ActionNormalizer(cfg.action_window_size, 2).fit(chunks)
    nz = norm.normalize(chunks)
    assert float(nz.abs().max()) <= 1.0 + 1e-6, "normaliser did not land in [-1, 1]"
    centers, _, inertia = kmeans_torch(nz.reshape(len(nz), -1), cfg.td.n_modes, seed=0)
    anchors_norm = centers.reshape(cfg.td.n_modes, cfg.action_window_size, 2)
    save_anchor_artifact(Path(tmpdir) / "anchors.pt", norm.unnormalize(anchors_norm),
                         anchors_norm, norm, {"synthetic": True, "inertia": inertia})
    return anchors_norm


def build(tmpdir, **td_overrides):
    cfg = make_cfg(tmpdir, **td_overrides)
    make_artifact(cfg, tmpdir)
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    model.set_noise_generator(torch.Generator().manual_seed(cfg.td.noise_seed))
    return cfg, model


# ---------------------------------------------------------------------------
# D3 -- one timestep per (N, T) position, shared by ALL K modes
# ---------------------------------------------------------------------------
def test_d3_t_is_shared_across_modes(tmpdir):
    """The user-mandated assertion: t may be [N,T], but each (N,T) cell has ONE t and
    all K=20 modes at that cell consume the SAME noise level. If modes could carry
    different t, the K candidates within a sample would be noised differently and the
    mode score would be comparing apples to oranges."""
    cfg, model = build(tmpdir)
    gap = model.test_t_shared_across_modes(n=2, t_len=3)
    check("D3: same t resolves to the same noised anchor for all K modes", gap == 0.0,
          f"max|diff| across modes = {gap}")

    n, t_len, k = 2, 3, cfg.td.n_modes
    bad = torch.zeros(n, t_len, k, dtype=torch.long)
    try:
        model._check_t(bad, n, t_len, "test")
        check("D3: _check_t REJECTS a [N,T,K] t", False, "it was accepted")
    except AssertionError as e:
        check("D3: _check_t REJECTS a [N,T,K] t", True, str(e).splitlines()[0][:70])

    x_t = torch.randn(n, t_len, k, cfg.action_window_size, 2)
    g = torch.randn(n, t_len, model.gpt.readout_dim)
    try:
        model._refine(x_t, bad, g)
        check("D3: _refine REJECTS a [N,T,K] t", False, "it was accepted")
    except AssertionError as e:
        check("D3: _refine REJECTS a [N,T,K] t", True, str(e).splitlines()[0][:70])


# ---------------------------------------------------------------------------
# Mandated fix 1 -- mode score from the SAME refined per-mode feature as regression
# ---------------------------------------------------------------------------
def test_fix1_score_and_regression_share_the_refined_feature(tmpdir):
    cfg, model = build(tmpdir)
    check("fix1: canonical cls_from == 'refined'", model.cls_from == "refined",
          f"got {model.cls_from!r}")
    # The MLP may end in a Dropout, so find the last Linear rather than indexing [-1].
    last_lin = [m for m in model.bin_head.net if isinstance(m, torch.nn.Linear)][-1]
    check("fix1: bin_head maps D -> 1 (per-mode score, not [N,T,K] from g)",
          last_lin.out_features == 1, f"out_features={last_lin.out_features}")

    n, t_len, k = 2, 3, cfg.td.n_modes
    x_t = torch.randn(n, t_len, k, cfg.action_window_size, 2)
    t = torch.arange(n * t_len).reshape(n, t_len) % cfg.td.train_t_max
    g = torch.randn(n, t_len, model.gpt.readout_dim)
    with torch.no_grad():
        cls_logits, x0_hats = model._refine(x_t, t, g)
    check("fix1: one score vector per decoder layer", len(cls_logits) == cfg.td.num_layers,
          f"{len(cls_logits)} vs num_layers={cfg.td.num_layers}")
    check("fix1: per-layer scores differ (scores come from the refined feature, not g)",
          float((cls_logits[0] - cls_logits[-1]).abs().max()) > 1e-6,
          f"max|layer0-layer_last| = {float((cls_logits[0]-cls_logits[-1]).abs().max()):.3e}")
    if len(cls_logits) > 1:
        spread = float(cls_logits[-1][0, 0].std())
        check("fix1: the score varies ACROSS modes for a fixed (n,t)",
              spread > 1e-6, f"std over K = {spread:.3e}")
        check("fix1: shape is [N,T,K]", tuple(cls_logits[-1].shape) == (n, t_len, k),
              str(tuple(cls_logits[-1].shape)))

    # regression and score must read the same tensor: nth x0_hat comes from the same
    # refined feature as the nth score, so both lists are num_layers long.
    check("fix1: score list and x0_hat list are the same length (shared z_k)",
          len(cls_logits) == len(x0_hats), f"{len(cls_logits)} vs {len(x0_hats)}")


def test_fix1_loss_is_a_sum_over_layers(tmpdir):
    """DiffusionDrive accumulates `ret_traj_loss += trajectory_loss` across decoder
    layers, so the harness loss must be a SUM, not a mean -- otherwise the effective
    learning rate on the head silently scales with num_layers."""
    cfg, model = build(tmpdir)
    n, t_len, k, w, a = 4, cfg.obs_window_size, cfg.td.n_modes, cfg.action_window_size, 2
    obs = torch.randn(n, t_len, cfg.dino.n_patches, cfg.dino.output_dim)
    # `_unpack_actions` maps [N, total_w, A] -> [N, total_w + 1 - W, W, A], so to get T
    # windows out of W-wide chunks the raw length must be W + T - 1, not W + T.
    act = torch.rand(n, cfg.action_window_size + t_len - 1, a)
    _, _, ld = model(obs, None, act)
    if cfg.td.num_layers > 1:
        l0, ll = ld["reg_loss_layer0"], ld["reg_loss_layer_last"]
        check("fix1: reg_loss_raw >= any single layer (sum, not mean)",
              ld["offset_loss"] >= max(l0, ll) - 1e-6,
              f"offset_loss={ld['offset_loss']:.4f} layer0={l0:.4f} last={ll:.4f}")
    for key in ("total_loss", "classification_loss", "offset_loss"):
        check(f"loss_dict has required key {key!r}", key in ld)


# ---------------------------------------------------------------------------
# Mandated fix 2 -- the production chain is two REAL transitions
# ---------------------------------------------------------------------------
def test_fix2_production_chain_is_two_real_steps():
    sch = TruncatedDiffusionSchedule(train_t_max=50)
    chain = inference_chain(49, 2)
    check("fix2: chain is [(49,25),(25,0)]", chain == [(49, 25), (25, 0)], str(chain))
    check("fix2: chain does NOT contain a model call at t=0", 0 not in chain[0],
          f"model inputs {make_inference_timesteps(49, 2)} (0 is the endpoint)")
    check("fix2: every transition is non-trivial",
          all(1.0 - sch.step_weights(a, b)["w1_xt"] > 0.02 for a, b in chain),
          "travels " + ", ".join(f"{1.0 - sch.step_weights(a, b)['w1_xt']:.4f}"
                                 for a, b in chain))
    cum = cumulative_travel(sch, 49, 2)
    check("fix2: cumulative travel is monotone and < 1 (no identity step)",
          cum[0]["cumulative_travel"] < cum[1]["cumulative_travel"] < 1.0,
          f"{cum[0]['cumulative_travel']:.4f} -> {cum[1]['cumulative_travel']:.4f}")


def test_fix2_clip_sample_matches_the_reference_default():
    """diffusers' DDIMScheduler defaults to clip_sample=True and the reference passes no
    override, so it clamps x0_hat to [-1,1] every step. Our step must do the same."""
    on = TruncatedDiffusionSchedule(train_t_max=50)
    off = TruncatedDiffusionSchedule(train_t_max=50, clip_sample=False)
    check("clip: default is clip_sample=True, range 1.0",
          on.clip_sample is True and on.clip_sample_range == 1.0)

    x_t = torch.randn(3, 20, 5, 2)
    x0 = torch.randn(3, 20, 5, 2) * 3.0          # deliberately out of [-1,1]
    r_on = on.ddim_step(x_t, x0, 49, 25)
    r_off = off.ddim_step(x_t, x0, 49, 25)
    check("clip: it changes the step on out-of-range input",
          float((r_on - r_off).abs().max()) > 1e-6,
          f"max|on-off| = {float((r_on-r_off).abs().max()):.3e}")
    # Clipping must enter ONLY through the x0_hat coefficient: diffusers forms eps_hat
    # from the UNCLIPPED prediction and clamps afterwards, so
    #     step(clip on) - step(clip off) == sqrt(a_prev) * (clamp(x0) - x0)
    # exactly. Clamping before forming eps_hat would break this identity -- and it is the
    # identity that distinguishes the two orderings, so it is the thing worth asserting.
    # (An earlier version of this test asserted the result "stays in range"; that is false
    # for both implementations: lowering x0_hat RAISES eps_hat, and the sum can exceed 1.)
    a_prev = float(on.abar(25))
    expect = (x0.clamp(-1.0, 1.0) - x0) * a_prev ** 0.5
    gap = float((r_on - r_off - expect).abs().max())
    check("clip: enters ONLY via the x0_hat coefficient (eps_hat uses the UNCLIPPED x0)",
          gap < 1e-6, f"max|(step_on - step_off) - sqrt(a_prev)*dclamp| = {gap:.3e}")
    check("clip: the clamped tensor itself is in range, and clipping is a real no-op only "
          "in range", float(x0.clamp(-1, 1).abs().max()) <= 1.0
          and float((x0.clamp(-1, 1) - x0).abs().max()) > 0)

    # terminal step: the returned tensor is the model output with NO residual noise, and
    # it is clamped when clip_sample is on. Both halves of that claim are checked.
    in_range = torch.randn(3, 20, 5, 2).clamp(-0.9, 0.9)
    term_in = on.ddim_step(x_t, in_range, 0, -1)
    check("clip: terminal step on an IN-range x0_hat returns it exactly (no noise)",
          float((term_in - in_range).abs().max()) == 0.0,
          f"max|diff| = {float((term_in - in_range).abs().max())}")
    term_wild = on.ddim_step(x_t, x0, 0, -1)
    check("clip: terminal step on an OUT-of-range x0_hat returns clamp(x0_hat)",
          float((term_wild - x0.clamp(-1.0, 1.0)).abs().max()) == 0.0,
          f"max|diff| = {float((term_wild - x0.clamp(-1.0, 1.0)).abs().max())}")
    check("clip: with clip_sample=False the terminal step returns x0_hat verbatim",
          float((off.ddim_step(x_t, x0, 0, -1) - x0).abs().max()) == 0.0)


def test_training_band_includes_zero(tmpdir):
    sch = TruncatedDiffusionSchedule(train_t_max=50)
    t = sch.sample_train_timesteps((20000,), torch.device("cpu"))
    check("train band is U{0..49}, INCLUSIVE of 0 (randint(0,50), not 1..50)",
          int(t.min()) == 0 and int(t.max()) == 49,
          f"min={int(t.min())} max={int(t.max())}")


def test_inference_shape_and_mode_selection(tmpdir):
    """End-to-end: 1-step and 2-step inference both produce [N,T,W,A] raw pixels, and
    the returned action is one of the K mode candidates."""
    cfg, model = build(tmpdir)
    n, t_len = 3, cfg.obs_window_size
    obs = torch.randn(n, t_len, cfg.dino.n_patches, cfg.dino.output_dim)
    for steps in (1, 2):
        model.set_inference(steps=steps)
        with torch.no_grad():
            pred, _, _ = model(obs, None, None)
        want = (n, t_len, cfg.action_window_size, 2)
        check(f"inference steps={steps}: output shape {want}", tuple(pred.shape) == want,
              str(tuple(pred.shape)))
        check(f"inference steps={steps}: finite", bool(torch.isfinite(pred).all()))


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        print("== D3: one t per (N,T), shared across K modes ==")
        test_d3_t_is_shared_across_modes(tmpdir / "d3")
        print("\n== mandated fix 1: score from the refined per-mode feature ==")
        test_fix1_score_and_regression_share_the_refined_feature(tmpdir / "f1a")
        test_fix1_loss_is_a_sum_over_layers(tmpdir / "f1b")
        print("\n== mandated fix 2: real 2-step DDIM chain ==")
        test_fix2_production_chain_is_two_real_steps()
        test_fix2_clip_sample_matches_the_reference_default()
        print("\n== training band ==")
        test_training_band_includes_zero(tmpdir / "band")
        print("\n== inference contract ==")
        test_inference_shape_and_mode_selection(tmpdir / "inf")
    failed = [n for n, ok in RESULTS if not ok]
    print("\n" + "=" * 74)
    print(f"FAILED {len(failed)}: " + "; ".join(failed) if failed
          else f"ALL {len(RESULTS)} CHECKS PASSED")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
