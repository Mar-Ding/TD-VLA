"""P3 -- gradient reachability audit for the TD arm.

Why this exists separately from `train.py --smoke`:

The harness's own smoke checklist (`train.py:1319-1355`) names SEVEN parameters:
    gpt.wte, gpt.wpe, gpt.h[0].attn.c_attn, gpt.h[0].mlp.c_fc, gpt.ln_f,
    bin_head.net[0], offset_head.net[0]
That list predates the TD arm. The module this project actually adds -- `denoiser`, with
`anchor_encoder`, `time_mlp` and `blocks[*]` -- is **not in it**. So a TD run can print
`SMOKE PASS` while the entire denoiser is disconnected from the loss. The harness
checklist is not wrong, it is simply blind to the new module, and no amount of staring at
its output would reveal that.

This script instead walks EVERY parameter of the built policy and reports, per top-level
module, how many received a finite gradient. It is strictly stronger than the checklist:
   * it covers all modules, not seven names;
   * it distinguishes "no gradient" (possibly legitimate -- a module not on the loss path)
     from "non-finite gradient" (never legitimate);
   * it asserts the modules that MUST be reached are reached.

Hard failures: any non-finite gradient, or zero grads on gpt / denoiser / bin_head /
offset_head. Those four are on the training loss path by construction, so zero means the
head is not learning -- which is the exact silent failure this stage is for.

CPU-only and harmless: no GPU, no data, no checkpoint, no training.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from patch_policy.config import Config  # noqa: E402

# Modules that MUST receive gradient in one training step. Everything else is reported
# but not gated: a param with no gradient is only a bug if the loss path touches it.
REQUIRED = ("gpt", "denoiser", "bin_head", "offset_head")

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def main() -> int:
    cfg = Config.from_yaml(sys.argv[1] if len(sys.argv) > 1
                           else "configs/pusht_dino_p256_td.yaml")
    assert cfg.action_head == "td", f"not the TD arm: action_head={cfg.action_head}"
    torch.manual_seed(0)

    from patch_policy.train import build_model
    model = build_model(cfg)
    model.train()

    obs = torch.randn(2, cfg.obs_window_size, 256, 384)
    act = torch.rand(2, 2 * cfg.action_window_size - 1, 2)   # W+T-1 raw actions
    _, loss, ld = model(obs, None, act)

    print("\n== gradient reachability ==")
    check("loss is finite", torch.isfinite(loss).item(), f"loss={float(loss):.4f}")

    # NOTE: bin_head/offset_head may have Dropout as their last mlp entry, and the
    # harness indexes .net[0]; a plain zero_grad + backward is all that is needed here.
    model.zero_grad(set_to_none=True)
    loss.backward()

    groups = {}
    for n, p in model.named_parameters():
        top = n.split(".")[0]
        g = p.grad
        tot, have, bad = groups.get(top, (0, 0, 0))
        tot += 1
        if g is not None:
            have += 1
            if not torch.isfinite(g).all():
                bad += 1
        groups[top] = (tot, have, bad)

    print(f"\n  {'module':<16}{'params':>8}{'with grad':>11}{'non-finite':>12}")
    for top in sorted(groups):
        tot, have, bad = groups[top]
        flag = "" if have == tot else "   <- partial"
        print(f"  {top:<16}{tot:>8}{have:>11}{bad:>12}{flag}")

    any_bad = any(bad for _, _, bad in groups.values())
    check("no non-finite gradient anywhere", not any_bad,
          "" if not any_bad else "non-finite grad in "
          + ", ".join(t for t, (_, _, b) in groups.items() if b))

    print("\n== required modules are on the loss path ==")
    for top in REQUIRED:
        if top not in groups:
            check(f"{top} exists", False, "module absent from the model")
            continue
        tot, have, bad = groups[top]
        check(f"{top} receives gradient on every param", have == tot and bad == 0,
              f"{have}/{tot} params with grad, {bad} non-finite")

    # The denoiser is the module this project adds; break it down so a future reader can
    # see WHICH submodule is dead rather than just that something was.
    #
    # `n.split(".")[0]` is the submodule name (`anchor_encoder`, `time_mlp`, `blocks`).
    # The first version of this used index [1], which on `anchor_encoder.0.weight` yields
    # `"0"` -- so it grouped by LAYER INDEX, matched no parameters, and printed four
    # vacuous `0/0 PASS` lines. A check that passes while measuring nothing is the exact
    # failure mode this audit exists to catch, so the `len(ps) > 0` assertion below makes
    # an empty group a FAIL rather than a PASS.
    print("\n== denoiser submodules (the module this project adds) ==")
    subs = sorted({n.split(".")[0] for n, _ in model.denoiser.named_parameters()})
    check("denoiser exposes submodules to check", len(subs) > 0, str(subs))
    for sub in subs:
        ps = [(n, p) for n, p in model.denoiser.named_parameters()
              if n.split(".")[0] == sub]
        have = sum(p.grad is not None for _, p in ps)
        check(f"denoiser.{sub} reached ({have}/{len(ps)})",
              len(ps) > 0 and have == len(ps))

    failed = [n for n, ok in RESULTS if not ok]
    print("\n" + "=" * 74)
    print(f"FAILED {len(failed)}: " + "; ".join(failed) if failed
          else f"ALL {len(RESULTS)} CHECKS PASSED")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
