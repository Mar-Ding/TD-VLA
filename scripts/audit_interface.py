"""P0 -- interface audit: does the TD head satisfy the harness's duck-typing contract?

The TD arm is meant to change exactly ONE thing (the action head), so the burden of proof
is that everything the harness touches still behaves. This script checks the contract
points by CALLING them, not by reading them:

  A. file identity vs the sibling tree ("same file, modulo the package rename")
  B. `gpt` readout shape and `readout_dim` (256 on the clean line, NOT 512)
  C. `forward` signature + return triple parity with `BehaviorTransformer`
  D. `loss_dict` carries the three keys `train.py` indexes unconditionally
  E. `vqvae_is_fit` / VQ hooks / `vqvae` stub
  F. checkpoint round-trip, and no `_vqvae_model.*` key in a TD checkpoint
  G. rollout shape contract: pred[:, -1] -> [N,5,2]
  H. DINO frozen and the right width
  I. envs resolution on the server (`envs` from ~/patch_policy, `patch_policy` from ours)
  J. GPU2 rollout lock mutual exclusion -- DEFERRED to P3, see the note printed by I/J.

Checks A and I/J need the server / sibling tree; the rest are self-contained and CPU-only.

Usage:
    python scripts/audit_interface.py                 # local (A-H)
    python scripts/audit_interface.py --remote         # also I/J, must run on the server
"""
import argparse
import difflib
import inspect
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
# The pristine upstream clone used to sit at ROOT/base/patch_policy. On 2026-09-21 it
# became the repo-wide prototype at <repo>/Patch policy/upstream. Both locations are
# tried so this audit still runs in a checkout that predates the move; `base/` itself
# survives (it also holds the unrelated third-party DiffusionDrive clone).
_UPSTREAM_CANDIDATES = (
    ROOT.parent / "Patch policy" / "upstream",
    ROOT / "base" / "patch_policy",
)
BASE = next((p for p in _UPSTREAM_CANDIDATES if p.is_dir()), _UPSTREAM_CANDIDATES[0])
SIBLING = ROOT.parent / "sibling-project" / "sibling_pkg"
RESULTS = []

# Files that must be identical to the sibling tree after `s/sibling_pkg/patch_policy/g`.
SHARED = ("gpt.py", "encoder.py", "dataset.py", "vqvae.py", "vqbet.py",
          "coord_decoder.py", "adaln.py", "adaln_gate.py", "global_policy.py",
          "precompute.py", "precompute_cond.py", "qwen3vl_official.py")

# Files with an ENUMERATED, justified deviation, each with a BUDGET in changed lines
# (added + removed, from difflib -- NOT a positional line-by-line comparison, which
# reports every subsequent line as "changed" after a single insertion and so turns a
# 28-line diff into a 1105-line scare). The budget is the point: the deviation is
# allowed, but it must not grow silently.
EXPECT_DEVIATIONS = {
    "__init__.py": (20, "docstring rewritten to describe THIS project, not the fork"),
    "eval_rollout.py": (25, "(a) usage header ~/sibling_pkg -> ~/td_pusht; "
                            "(b) BASE overridable by $PATCH_POLICY_BASE, default unchanged"),
    "train.py": (40, "build_model dispatch (5 lines) + 2 annotation widenings + docstring. "
                     "The ONLY substantive harness change in this project"),
    "config.py": (130, "TDConfig dataclass + Config.td field; no existing field altered"),
}

# NEW modules this project adds. Provenance matters: `normalizer.py` is NOT a copy of
# anything in the sibling tree (the sibling has no normaliser at all) -- it is a
# reduction of the BASE repo's original, so comparing it against the sibling would
# invent a failure.
NEW_FILES = {
    "normalizer.py": (BASE / "utils" / "normalizer.py",
                      "reduction of the base repo's _fit/_normalize/limits mode: drops "
                      "zarr, numpy and the dict-of-fields machinery"),
    "anchors.py": (None, "no upstream: pure-torch k-means over PushT action chunks"),
    "diffusion.py": (None, "no upstream: self-contained truncated DDIM (wemm-p1 has no "
                           "diffusers)"),
    "td_head.py": (None, "no upstream: TruncatedDiffusionHead + TruncatedDiffusionPolicy"),
}


def changed_lines(a_text, b_text):
    """Added + removed lines between two texts, ignoring pure position shifts."""
    a = a_text.splitlines()
    b = b_text.splitlines()
    return sum(1 for line in difflib.unified_diff(a, b, lineterm="", n=0)
               if line[:1] in "+-" and not line.startswith(("+++", "---")))


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def check_a_file_identity(required=True):
    print("\n== A. file identity vs the sibling tree (modulo the package rename) ==")
    mine = ROOT / "patch_policy"

    if not SIBLING.is_dir():
        # Only a FAIL where the sibling is expected to exist (the local work tree).
        # On the server it does not, and calling that a failure would be an alarm
        # about the absence of a comparison baseline, not about this code.
        check("A: sibling tree present", not required,
              f"no sibling tree at {SIBLING} (expected off-local; "
              f"pass --sibling to point at one)")
        return

    for name in SHARED:
        src, dst = SIBLING / name, mine / name
        if not src.is_file():
            check(f"A: {name} exists in sibling", False, str(src))
            continue
        # Normalise the rename and the import lines it touches, then compare exactly.
        text = src.read_text(encoding="utf-8", errors="replace").replace(
            "sibling_pkg", "patch_policy")
        check(f"A: {name} is identical modulo the package rename",
              text == dst.read_text(encoding="utf-8", errors="replace"))

    print("  -- enumerated deviations (budget = changed lines, difflib) --")
    for name, (budget, why) in EXPECT_DEVIATIONS.items():
        src, dst = SIBLING / name, mine / name
        if not (src.is_file() and dst.is_file()):
            check(f"A: {name} present on both sides", False)
            continue
        text = src.read_text(encoding="utf-8", errors="replace").replace(
            "sibling_pkg", "patch_policy")
        n = changed_lines(text, dst.read_text(encoding="utf-8", errors="replace"))
        check(f"A: {name} deviates within budget ({n}/{budget} changed lines)",
              n <= budget, why)

    print("  -- new modules (no sibling counterpart; provenance checked instead) --")
    for name, (upstream, why) in NEW_FILES.items():
        dst = mine / name
        if not dst.is_file():
            check(f"A: {name} exists", False, str(dst))
            continue
        check(f"A: {name} exists and is not a sibling copy",
              not (SIBLING / name).is_file(), why)
        if upstream is not None:
            check(f"A: {name} upstream original present ({upstream.parent.name}/"
                  f"{upstream.name})", upstream.is_file(), str(upstream))


def make_cfg(tmpdir, action_head="td", **td_overrides):
    from patch_policy.config import Config, TDConfig
    cfg = Config()
    cfg.encoder_name = "dino"
    cfg.dino.postprocess = None
    cfg.dino.n_patches = 256
    cfg.action_head = action_head
    cfg.obs_window_size = 5
    cfg.action_window_size = 5
    cfg.policy.native_parity = True          # the `clean` line
    cfg.policy.n_layer = 2
    cfg.policy.n_head = 8
    cfg.policy.n_embd = 512
    cfg.td = TDConfig(anchor_artifact=str(Path(tmpdir) / "anchors.pt"),
                      n_modes=20, **td_overrides)
    return cfg


def make_artifact(cfg, tmpdir):
    from patch_policy.anchors import kmeans_torch, save_anchor_artifact
    from patch_policy.normalizer import ActionNormalizer
    g = torch.Generator().manual_seed(0)
    n = 4000
    a = torch.stack([torch.randint(12, 512, (n,), generator=g).float(),
                     torch.randint(25, 512, (n,), generator=g).float()], dim=-1)
    chunks = a.reshape(-1, cfg.action_window_size, 2)[: n // cfg.action_window_size]
    norm = ActionNormalizer(cfg.action_window_size, 2).fit(chunks)
    nz = norm.normalize(chunks)
    centers, _, inertia = kmeans_torch(nz.reshape(len(nz), -1), cfg.td.n_modes, seed=0)
    anchors_norm = centers.reshape(cfg.td.n_modes, cfg.action_window_size, 2)
    save_anchor_artifact(Path(tmpdir) / "anchors.pt", norm.unnormalize(anchors_norm),
                         anchors_norm, norm, {"synthetic": True, "inertia": inertia})


def check_b_to_h(tmpdir):
    from patch_policy.train import build_model
    cfg = make_cfg(tmpdir)
    make_artifact(cfg, tmpdir)
    torch.manual_seed(0)
    model = build_model(cfg).eval()

    print("\n== B. trunk readout ==")
    dummy = torch.randn(2, 5, 256, 384)
    with torch.no_grad():
        g = model.gpt(dummy, None)
    check("B: gpt(dummy[2,5,256,384]) -> [2,5,256]", tuple(g.shape) == (2, 5, 256),
          str(tuple(g.shape)))
    check("B: readout_dim == 256 (clean line: lm_head Linear(512->256))",
          model.gpt.readout_dim == 256, f"got {model.gpt.readout_dim}")
    check("B: the head consumes readout_dim, not a hard-coded 512",
          model.gpt.readout_dim == cfg.policy.n_embd // 2,
          f"readout_dim={model.gpt.readout_dim}, n_embd={cfg.policy.n_embd}")

    print("\n== C. forward contract parity with BehaviorTransformer ==")
    from patch_policy.vqbet import BehaviorTransformer
    # `model.forward` is a BOUND method so its signature omits `self`; the class
    # attribute is a plain function and still has it. Compare like with like.
    td_params = list(inspect.signature(model.forward).parameters)
    bt_params = list(inspect.signature(BehaviorTransformer.forward).parameters)
    bt_params = [p for p in bt_params if p != "self"]
    check("C: parameter names match BehaviorTransformer", td_params == bt_params,
          f"{td_params} vs {bt_params}")
    # Defaults too: the harness calls positionally AND by keyword depending on the path.
    td_def = inspect.signature(model.forward).parameters
    bt_def = inspect.signature(BehaviorTransformer.forward).parameters
    same_def = all(td_def[p].default == bt_def[p].default for p in bt_params)
    check("C: parameter defaults match BehaviorTransformer", same_def)
    obs = torch.randn(4, 5, 256, 384)
    act = torch.rand(4, 9, 2)
    out = model(obs, None, act)
    check("C: forward returns a 3-tuple", isinstance(out, tuple) and len(out) == 3,
          f"len={len(out) if isinstance(out, tuple) else 'n/a'}")
    pred, loss, ld = out
    check("C: predicted_action is [N,5,5,2]", tuple(pred.shape) == (4, 5, 5, 2),
          str(tuple(pred.shape)))
    check("C: loss is a scalar tensor with grad", loss.ndim == 0 and loss.requires_grad)
    check("C: inference call (action_seq=None) also returns a 3-tuple",
          isinstance(model(obs, None, None), tuple)
          and len(model(obs, None, None)) == 3)

    print("\n== D. loss_dict keys train.py indexes unconditionally ==")
    for k in ("total_loss", "classification_loss", "offset_loss"):
        check(f"D: loss_dict['{k}'] present and finite", k in ld
              and torch.isfinite(torch.tensor(float(ld[k]))), f"{ld.get(k)}")

    print("\n== E. VQ surface ==")
    check("E: vqvae_is_fit is True (no tokenizer to fit)", model.vqvae_is_fit is True)
    try:
        model.vqvae_is_fit = True          # train.py --resume historically assigns this
        check("E: assigning vqvae_is_fit is absorbed, not an AttributeError", True)
    except Exception as e:                 # noqa: BLE001
        check("E: assigning vqvae_is_fit is absorbed, not an AttributeError", False, repr(e))
    check("E: vqvae stub has .state_dict()/.training/.parameters()",
          callable(model.vqvae.state_dict) and isinstance(model.vqvae.training, bool)
          and callable(model.vqvae.parameters))
    try:
        model._maybe_fit_vq(force=True)
        check("E: --export-shared-vq refuses on the TD arm", False, "it did not raise")
    except SystemExit as e:
        check("E: --export-shared-vq refuses on the TD arm", True, str(e)[:60])

    print("\n== F. checkpoint round-trip ==")
    sd = model.state_dict()
    vq_keys = [k for k in sd if "_vqvae_model" in k or k.startswith("vqvae.")]
    check("F: no VQ-tokenizer keys in a TD state_dict", not vq_keys, str(vq_keys[:4]))
    tmp = Path(tmpdir) / "ckpt.pt"
    torch.save({"model": sd}, tmp)
    m2 = build_model(cfg)
    m2.load_state_dict(torch.load(tmp, map_location="cpu")["model"])
    same = all(torch.equal(sd[k], m2.state_dict()[k]) for k in sd)
    check("F: save -> load -> identical state_dict", same)

    print("\n== G. rollout shape contract ==")
    with torch.no_grad():
        p, _, _ = m2.eval()(obs, None, None)
    last = p[:, -1]
    check("G: pred[:, -1] -> [N,5,2]", tuple(last.shape) == (4, 5, 2), str(tuple(last.shape)))
    stacked = torch.stack([last] * 5, dim=0)          # the W=5 temporal ensemble
    check("G: temporal ensemble [W,N,5,2] -> [:, 0] is [N,5,2]",
          tuple(stacked[0].shape) == (4, 5, 2))

    print("\n== H. encoder frozen ==")
    from patch_policy.encoder import build_encoder
    enc, _ = build_encoder(cfg)
    n_grad = sum(p.requires_grad for p in enc.parameters())
    check("H: all encoder params require_grad=False", n_grad == 0, f"{n_grad} trainable")
    check("H: n_patches=256, output_dim=384",
          (enc.n_patches, enc.output_dim) == (256, 384),
          f"({enc.n_patches}, {enc.output_dim})")



def check_i_envs():
    print("\n== I. envs / package resolution on the server ==")
    # Import the module that ACTUALLY does the resolution, rather than re-deriving the
    # sys.path rule here. If `eval_rollout.py` ever stops adding BASE, this check fails
    # -- which is the point; a reimplementation would happily keep passing.
    import patch_policy.eval_rollout  # noqa: F401  (side effect: inserts BASE on sys.path)
    import envs
    import patch_policy

    envs_dir = str(Path(envs.__file__).resolve().parent)
    pkg_dir = str(Path(patch_policy.__file__).resolve().parent)
    base = os.environ.get("PATCH_POLICY_BASE", os.path.expanduser("~/patch_policy"))
    print(f"    BASE         -> {base}")
    print(f"    envs         -> {envs_dir}")
    print(f"    patch_policy -> {pkg_dir}")

    check("I: envs resolves under BASE (the read-only upstream)",
          envs_dir.startswith(str(Path(base).resolve()) + os.sep), envs_dir)
    # patch_policy must resolve from the cwd (our tree), NOT from BASE. This is the
    # no-shadowing claim; it holds only because BASE has no patch_policy/ subdir.
    check("I: patch_policy resolves from the cwd, not from BASE",
          not pkg_dir.startswith(str(Path(base).resolve()) + os.sep), pkg_dir)
    check("I: BASE contains no patch_policy/ subdir (so it cannot shadow ours)",
          not (Path(base) / "patch_policy").exists(), str(Path(base) / "patch_policy"))
    check("I: this project is a different tree from BASE", pkg_dir != base, pkg_dir)


def main():
    global SIBLING
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true",
                    help="also run the server-only checks (must be ON the server)")
    ap.add_argument("--sibling", default=None,
                    help="path to the sibling_pkg tree to compare against (default: the "
                         "local work tree; absent on the server, where A is skipped)")
    args = ap.parse_args()
    if args.sibling:
        SIBLING = Path(args.sibling).resolve()
    torch.set_num_threads(2)

    check_a_file_identity(required=not args.remote)
    with tempfile.TemporaryDirectory() as td:
        check_b_to_h(Path(td))
    print("\n== I/J. server-only checks ==")
    if args.remote:
        check_i_envs()
    else:
        print("    skipped (local run); pass --remote on the server")
    print("    J (GPU2 lock mutual exclusion) is DEFERRED to P3: it needs")
    print("    rollout_locked_gpu2.sh, which is written at the P3 rollout stage. Until")
    print("    then there is no lock to test, and claiming it passes would be fiction.")

    failed = [n for n, ok in RESULTS if not ok]
    print("\n" + "=" * 74)
    print(f"FAILED {len(failed)}: " + "; ".join(failed) if failed
          else f"ALL {len(RESULTS)} CHECKS PASSED")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
