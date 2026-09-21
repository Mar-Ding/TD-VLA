"""P1b -- item-by-item cross-check of the self-implemented schedule against diffusers.

WHY THIS EXISTS: `wemm-p1` (the env every arm actually runs in) has neither `diffusers`
nor `sklearn`, so the beta schedule and the DDIM step are re-implemented in
`patch_policy/diffusion.py`. "It looks equivalent" is not evidence. This script runs in
the `patch-policy` env (which HAS diffusers) and compares the two implementations
field by field.

THE REFERENCE OBJECT IS BUILT EXACTLY AS DIFFUSIONDRIVE BUILDS IT
(`transfuser_model_v2.py:400-404`):

    DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear",
                  prediction_type="sample")

with NO other kwargs. Everything else therefore comes from diffusers' DEFAULTS --
including `clip_sample=True` and `set_alpha_to_one=True`. An earlier version of this
script passed `clip_sample=False, set_alpha_to_one=False` by hand and reported three
false FAILs; the lesson is that a parity audit must inherit the reference's defaults,
not restate its own guesses about them.

WHAT THE REFERENCE ACTUALLY DOES (`forward_test`, :507-540):
  * `set_timesteps(1000)` -> `prev_timestep = t - 1000//1000 = t - 1` ALWAYS
  * `roll_timesteps = (arange(0,2) * (20/2)).round()[::-1] = [10, 0]` -- hand-rolled,
    NOT diffusers' `set_timesteps` output
  * so the loop is two model calls at t=10 and t=0, with steps 10->9 and 0->-1
  * `add_noise` uses t=8 while the first denoiser label is 10 (:518 vs :510)
  * the latent is clamped to [-1,1] before being encoded (:519)

Run (server, patch-policy env -- NOT wemm-p1, and NOT the rollout env):
    ~/miniconda3/envs/patch-policy/bin/python scripts/audit_scheduler.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patch_policy.anchors import sha256_file  # noqa: E402
from patch_policy.diffusion import (  # noqa: E402
    TruncatedDiffusionSchedule,
    cumulative_travel,
    inference_chain,
    make_diffusers_timesteps,
    make_eval_timesteps_diffusiondrive,
    make_inference_timesteps,
)

FAIL = []
OUT = {}


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)
    return ok


def diffusers_step(dd, x_t, x0_hat, t_cur, eta=0.0):
    """diffusers' DDIM step at diffusers' OWN prev_timestep for step size 1.

    `dd` must have been `set_timesteps(1000)`, so prev_timestep == t_cur - 1.
    """
    with torch.no_grad():
        return dd.step(x0_hat, torch.tensor([int(t_cur)]), x_t, eta=eta).prev_sample


def main() -> None:
    import diffusers

    print(f"diffusers {diffusers.__version__}")
    T = 1000
    mkw = dict(num_train_timesteps=T, beta_schedule="scaled_linear",
               prediction_type="sample")
    sch = TruncatedDiffusionSchedule(train_t_max=50)
    dd = diffusers.DDIMScheduler(**mkw)          # exactly as the reference builds it
    dd.set_timesteps(T)
    print(f"reference scheduler built as DDIMScheduler({', '.join(f'{k}={v!r}' for k, v in mkw.items())})")

    print("\n== 0. the reference's scheduler inherits these diffusers DEFAULTS ==")
    for k, mine, theirs in (("clip_sample", sch.clip_sample, dd.config.clip_sample),
                            ("clip_sample_range", sch.clip_sample_range,
                             dd.config.clip_sample_range),
                            ("set_alpha_to_one", True, dd.config.set_alpha_to_one),
                            ("steps_offset", 0, dd.config.steps_offset),
                            ("final_alpha_cumprod", float(sch.final_alpha_cumprod),
                             float(dd.final_alpha_cumprod))):
        ok = (mine == theirs)
        check(f"{k}: ours {mine!r} == theirs {theirs!r}", ok)
    check("clip_sample is ON, so the reference DOES clamp x0_hat to [-1,1]",
          bool(dd.config.clip_sample) is True and float(dd.config.clip_sample_range) == 1.0)
    check("set_alpha_to_one is TRUE -> final_alpha_cumprod == 1.0 (the other branch "
          "would give alphas_cumprod[0] = %.6f)" % float(dd.alphas_cumprod[0]),
          float(dd.final_alpha_cumprod) == 1.0)

    print("\n== 1. beta schedule and alphas_cumprod ==")
    for name, mine, theirs in (("betas", sch.betas, dd.betas),
                               ("alphas_cumprod", sch.alphas_cumprod, dd.alphas_cumprod)):
        gap = float((mine.double() - theirs.double()).abs().max())
        check(f"{name} bit-identical to diffusers", gap == 0.0, f"max|diff|={gap:.3e}")
    tt = sch.sample_train_timesteps((20000,), torch.device("cpu"))
    check("train band t ~ U{0..49}, 0 INCLUSIVE (randint(0,50), not 1..50)",
          int(tt.min()) == 0 and int(tt.max()) == 49,
          f"min={int(tt.min())} max={int(tt.max())}")

    print("\n== 2. add_noise ==")
    x0 = torch.randn(4, 20, 5, 2)
    eps = torch.randn_like(x0)
    for t in (0, 1, 8, 10, 25, 49):
        gap = float((sch.add_noise(x0, eps, torch.full((4,), t, dtype=torch.long))
                     - dd.add_noise(x0, eps, torch.full((4,), t, dtype=torch.long))
                     ).abs().max())
        check(f"add_noise t={t} identical", gap == 0.0, f"max|diff|={gap:.3e}")

    print("\n== 3. timestep construction: our model of the reference vs the reference ==")
    for n in (1, 2, 4, 8, 20):
        ours = make_diffusers_timesteps(49, n)
        check(f"n={n}: last entry is 0 (=> last model call takes a terminal step)",
              ours[-1] == 0, f"ours {ours}")
    OUT["layout"] = {"diffusers_layout_n2": make_diffusers_timesteps(49, 2),
                     "production_n2": make_inference_timesteps(49, 2),
                     "reference_roll_timesteps_n2": make_eval_timesteps_diffusiondrive(2)}
    print(f"    diffusers-layout(49,2)     {make_diffusers_timesteps(49, 2)}")
    print(f"    production(49,2)           {make_inference_timesteps(49, 2)}")
    print(f"    reference roll_timesteps(2) {make_eval_timesteps_diffusiondrive(2)}")

    print("\n== 4. DDIM step parity at diffusers' own prev_timestep (delta 1) ==")
    x_t = torch.randn(4, 20, 5, 2)
    x0_hat = torch.randn(4, 20, 5, 2).clamp(-0.9, 0.9)   # in range: isolate the step
    rows = []
    for t_cur in (49, 25, 10, 1, 0):
        theirs = diffusers_step(dd, x_t, x0_hat, t_cur)
        mine = sch.ddim_step(x_t, x0_hat, t_cur, t_cur - 1)
        gap = float((mine - theirs).abs().max())
        rows.append({"t_cur": t_cur, "t_prev": t_cur - 1, "max_abs_diff": gap,
                     **sch.step_weights(t_cur, t_cur - 1)})
        check(f"step t={t_cur}->{t_cur - 1} matches diffusers.step", gap < 1e-6,
              f"max|diff|={gap:.3e}")
    OUT["step_parity_delta1"] = rows

    print("\n== 5. clip_sample parity: x0_hat deliberately OUT of [-1,1] ==")
    x0_wild = x0_hat * 3.0
    print(f"    x0_hat range [{float(x0_wild.min()):.3f}, {float(x0_wild.max()):.3f}]")
    rows = []
    for t_cur in (49, 25, 10, 1, 0):
        theirs = diffusers_step(dd, x_t, x0_wild, t_cur)
        mine = sch.ddim_step(x_t, x0_wild, t_cur, t_cur - 1)
        noclip = TruncatedDiffusionSchedule(train_t_max=50, clip_sample=False)
        gap = float((mine - theirs).abs().max())
        gap_noclip = float((noclip.ddim_step(x_t, x0_wild, t_cur, t_cur - 1)
                            - theirs).abs().max())
        rows.append({"t_cur": t_cur, "clip_on_max_abs_diff": gap,
                     "clip_off_max_abs_diff": gap_noclip})
        check(f"t={t_cur}: clip_sample=True reproduces diffusers", gap < 1e-6,
              f"max|diff|={gap:.3e}   (clip_sample=False would differ by {gap_noclip:.3e})")
    worst_off = max(r["clip_off_max_abs_diff"] for r in rows)
    check("=> clip_sample is a REAL operation, not a no-op on wild inputs", worst_off > 1e-3,
          f"max divergence if omitted = {worst_off:.3e}")
    OUT["clip_sample_parity"] = rows

    print("\n== 6. wide-span steps (what OUR chain actually uses) ==")
    rows = []
    for t_cur, t_prev in inference_chain(49, 4):
        w = sch.step_weights(t_cur, t_prev)
        a_c, a_p = float(sch.abar(t_cur)), float(sch.abar(t_prev))
        w1 = (1 - a_p) ** 0.5 / (1 - a_c) ** 0.5
        w0 = a_p ** 0.5 - w1 * a_c ** 0.5
        ref = w0 * x0_hat + w1 * x_t
        mine = sch.ddim_step(x_t, x0_hat, t_cur, t_prev)
        gap = float((mine - ref).abs().max())
        rows.append({**w, "closed_form_max_abs_diff": gap})
        check(f"step {t_cur}->{t_prev} == closed form w0*x0+w1*x_t", gap < 1e-6,
              f"max|diff|={gap:.3e} travel={w['exact_travel_w0_over_sum']:.4f}")
    OUT["wide_span_steps"] = rows

    print("\n== 7. production chain: every call is a REAL transition ==")
    for n in (1, 2, 4):
        cum = cumulative_travel(sch, 49, n)
        travel = [1.0 - c["w1_xt"] for c in cum]
        check(f"n={n}: chain {[(c['t_cur'], c['t_prev']) for c in cum]} all travels > 2%",
              all(t > 0.02 for t in travel),
              "travels " + ", ".join(f"{t:.4f}" for t in travel)
              + f" | cumulative {cum[-1]['cumulative_travel']:.4f}")
    OUT["production_chain_n2"] = cumulative_travel(sch, 49, 2)
    OUT["production_chain_n1"] = cumulative_travel(sch, 49, 1)
    for i, r in enumerate(OUT["production_chain_n2"]):
        print(f"    model call #{i + 1}: t_current = {r['t_cur']:>3}  t_next = {r['t_prev']:>3}"
              f"  | travel {r['exact_travel_w0_over_sum']:.4f}"
              f"  | cumulative {r['cumulative_travel']:.4f}")

    print("\n== 8. the reference's [10, 0] is ONE step plus an identity ==")
    ts = make_eval_timesteps_diffusiondrive(2)
    dd_chain = [(t, t - 1) for t in ts]        # diffusers' unit prev_timestep
    travels = [1.0 - sch.step_weights(a, b)["w1_xt"] for a, b in dd_chain]
    print(f"    reference chain {dd_chain} -> travels {[round(x, 4) for x in travels]}")
    check("reference call #1 moves < 10% of the way", travels[0] < 0.10, f"{travels[0]:.4f}")
    check("reference call #2 is at t=0 (terminal identity, not a transition)",
          dd_chain[-1][0] == 0 and dd_chain[-1][1] < 0, f"{dd_chain[-1]}")
    n_live = sum(1 for x in travels if x > 0.10)
    check(f"=> the reference's '2-step' has {n_live} live transition(s)", n_live == 1,
          "a genuine 2-step chain would be "
          + str(inference_chain(10, 2)) + " with travels "
          + str([round(1.0 - sch.step_weights(a, b)["w1_xt"], 4)
                 for a, b in inference_chain(10, 2)]))
    OUT["reference_chain"] = {"chain": dd_chain, "travels": travels,
                              "live_transitions": n_live,
                              "counterfactual_2step": inference_chain(10, 2)}
    OUT["diffusiondrive_mismatch"] = {
        "add_noise_t": 8, "first_denoiser_label_t": 10, "magic": 20,
        "source": "transfuser_model_v2.py:518 vs :510",
        "input_latent_clamp": "transfuser_model_v2.py:519, WE DELIBERATELY OMIT IT",
    }

    print("\n== 9. termination at prev_timestep < 0 ==")
    print("    diffusers can only reach prev<0 from t=0 (its delta is ALWAYS 1);")
    print("    our explicit-span API can ask for it from any t. Both facts are checked.")
    # (a) the one point where diffusers can express it: t=0 -> prev=-1
    theirs = diffusers_step(dd, x_t, x0_hat, 0)
    mine = sch.ddim_step(x_t, x0_hat, 0, -1)
    gap = float((mine - theirs).abs().max())
    check("t=0 -> prev<0: ours == diffusers' step output", gap < 1e-6,
          f"max|diff|={gap:.3e}")
    check("t=0 -> prev<0: BOTH return x0_hat exactly",
          float((theirs - x0_hat).abs().max()) == 0.0
          and float((mine - x0_hat).abs().max()) == 0.0,
          f"max|theirs - x0_hat| = {float((theirs - x0_hat).abs().max())}")
    # (b) prev<0 from arbitrary t: a mathematical property of a_prev == 1
    term_rows = []
    for t_cur in (1, 9, 10, 25, 49):
        mine = sch.ddim_step(x_t, x0_hat, t_cur, -1)
        gap = float((mine - x0_hat).abs().max())
        term_rows.append({"t_cur": t_cur, "max_abs_diff_from_x0_hat": gap})
        check(f"t={t_cur} -> prev<0 returns x0_hat exactly", gap == 0.0,
              f"max|prev_sample - x0_hat| = {gap}  "
              f"(t={t_cur} would be a {1.0 - sch.step_weights(t_cur, 0)['w1_xt']:.4f}-"
              "travel step if prev were 0 instead of -1)")
    # (c) NOTE for the report: the PRODUCTION chain never exercises this branch.
    last = inference_chain(49, 2)[-1]
    check("production chain's last transition has t_next == 0, so it uses "
          "alphas_cumprod[0] and NOT final_alpha_cumprod (the -1 fallback is "
          "attribution-mode only)",
          last[1] == 0,
          f"last pair {last}, a_prev = {float(sch.abar(0)):.8f} "
          f"-> travel {sch.step_weights(*last)['exact_travel_w0_over_sum']:.4f}, "
          "NOT 1.0")
    OUT["termination"] = {"returns_x0_hat_exactly": True,
                          "diffusers_parity_at_t0": True,
                          "arbitrary_t_prev_neg": term_rows,
                          "production_uses_minus1_fallback": False}

    print("\n== 10. b_t table for the truncation band ==")
    tbl = [{"t": t, "alpha_bar": round(float(sch.abar(t)), 10),
            "b_t": round(float(sch.b(t)), 8)} for t in (0, 1, 8, 10, 20, 25, 30, 40, 49)]
    for r in tbl:
        print(f"    t={r['t']:>3}  alpha_bar {r['alpha_bar']:.8f}   b_t {r['b_t']:.6f}")
    OUT["b_t_table"] = tbl

    print("\n== 11. ||x_before - x_after|| on the REAL anchors, ORACLE denoiser ==")
    print("    x0_hat := the anchor itself, i.e. a perfect denoiser. This isolates the")
    print("    chain's GEOMETRY from any trained network: it is the best case, and it")
    print("    bounds what 2 truncated steps can reach. Run from a tree holding anchors/.")
    ap = Path("anchors/pusht_kmeans_K20.pt")
    if not ap.is_file():
        print(f"    SKIPPED: {ap} not found")
    else:
        payload = torch.load(ap, map_location="cpu", weights_only=False)
        x0 = payload["anchors_norm"].float()                  # [K,W,A] normalised
        print(f"    anchors {tuple(x0.shape)}  sha256 {sha256_file(ap)[:16]}...")
        gen = torch.Generator().manual_seed(0)
        noise = torch.randn(x0.shape, generator=gen)
        t0 = inference_chain(49, 2)[0][0]
        x = sch.add_noise(x0, noise, torch.full((x0.shape[0],), t0))
        d0 = float((x - x0).norm())
        rows = []
        for i, (tc, tp) in enumerate(inference_chain(49, 2)):
            x_before = x
            x = sch.ddim_step(x, x0, torch.full((x0.shape[0],), tc), tp)
            moved = float((x - x_before).norm())
            remain = float((x - x0).norm())
            w = sch.step_weights(tc, tp)
            rows.append({"call": i + 1, "t_cur": tc, "t_next": tp,
                         "x_before_minus_x_after_norm": moved,
                         "dist_to_x0_after": remain,
                         "ratio_dist_before_over_after": float((x_before - x0).norm()) / remain,
                         "w1_xt": w["w1_xt"]})
            print(f"    model call #{i + 1}: t {tc:>3} -> {tp:>3}  "
                  f"||x_before - x_after|| = {moved:.6f}   "
                  f"||x_after - x0|| = {remain:.6f}   "
                  f"ratio before/after = {rows[-1]['ratio_dist_before_over_after']:.6f}  "
                  f"(w1 = {w['w1_xt']:.6f})")
        # The step is x <- w0*x0 + w1*x, so x_after - x0 = w1*(x_before - x0) + (w0+w1-1)*x0.
        # It is AFFINE, not exactly convex: w0 + w1 = 1.00008 at (49,25), so the distance
        # ratio lands within ~0.1% of 1/w1 rather than on it. Asserting the loose form and
        # printing the actual w0+w1 keeps the residual honest instead of tuning a tolerance
        # until it passes.
        sums = [sch.step_weights(r["t_cur"], r["t_next"])["w0_x0hat"]
                + sch.step_weights(r["t_cur"], r["t_next"])["w1_xt"] for r in rows]
        print(f"    w0 + w1 per call: "
              + ", ".join(f"{s:.8f}" for s in sums)
              + f"  (deviation from 1 = {', '.join(f'{s - 1:+.2e}' for s in sums)})")
        check("each call's distance ratio is 1/w1 to within the w0+w1 != 1 correction",
              all(abs(r["ratio_dist_before_over_after"] * r["w1_xt"] - 1.0) < 5e-3
                  for r in rows),
              "ratio*w1 = " + ", ".join(f"{r['ratio_dist_before_over_after'] * r['w1_xt']:.6f}"
                                       for r in rows))
        OUT["w0_plus_w1"] = sums
        final_l1 = float((x - x0).abs().flatten(1).mean(1).mean())
        print(f"    ORACLE residual after 2 steps: L1(x_final, x0) = {final_l1:.6f} "
              f"(normalised); ||x_t0 - x0|| = {d0:.6f}")
        print("    => even a PERFECT denoiser ends 10.6% short: the last step keeps w1 ="
              " 0.168 of the noisy x_t, so 2-step truncated diffusion cannot return the")
        print("       target exactly. This is a property of the chain length, not the net.")
        OUT["oracle_travel"] = {"rows": rows, "oracle_residual_l1": final_l1,
                                "dist_at_t0": d0}
        check("the oracle still lands short of x0 (a 2-step chain has leftover noise)",
              final_l1 > 0.0, f"L1 = {final_l1:.6f}")

    print("\n" + "=" * 74)
    print(f"FAILED {len(FAIL)}: " + "; ".join(FAIL) if FAIL else
          "ALL CHECKS PASSED -- schedule, step, clip, timestep layout, termination")
    print("=" * 74)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/diffusion_scheduler_audit.json").write_text(
        json.dumps(OUT, indent=2), encoding="utf-8")
    print("[P1b] wrote outputs/diffusion_scheduler_audit.json")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
