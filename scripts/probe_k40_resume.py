"""Is the K=40 arm's ep40 checkpoint resumable? Evidence, not opinion.

The question (user, 2026-09-18): 「先明确我们是否可以续训之前K=40的ckpt，在已经到达了40epoch的时候」

`train.py --resume` has four hard gates (resume_optimizer, train.py:212-275) plus two
model-level gates (trainable_param_names, vqvae_is_fit). This script answers each one
by INSPECTING THE ACTUAL FILE rather than by reading the writer's intent, because the
only failure mode that matters is "the file on disk lacks a key the writer promised".

Run on the server:
    cd ~/td_pusht && python scripts/probe_k40_resume.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path.home() / "td_pusht"
RUN = ROOT / "outputs" / "ksweep_k40" / "seed0"

def hr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)

hr("1. what is actually in the K=40 run directory")
if not RUN.is_dir():
    print(f"[FATAL] {RUN} does not exist")
    sys.exit(1)
for p in sorted(RUN.iterdir()):
    print(f"  {p.stat().st_size:>13,}  {p.name}  (mtime {p.stat().st_mtime:.0f})")

# The label in ckpt_epoch{N}_ is the 1-based epoch (train.py:1278 `e = epoch + 1`),
# so the highest N is the last epoch trained. Anything above 40 would mean someone
# already resumed this arm once.
ckpts = sorted(RUN.glob("ckpt_epoch*_step*.pt"))
print(f"\n  {len(ckpts)} ckpt(s); epoch labels: "
      f"{[int(p.name.split('_')[1][5:]) for p in ckpts]}")
TARGET = RUN / "ckpt_epoch40_step115680.pt"
if not TARGET.is_file():
    cands = [p for p in ckpts if p.name.startswith("ckpt_epoch40_")]
    if not cands:
        print("[FATAL] no ckpt with epoch label 40 -- nothing to resume from")
        sys.exit(1)
    TARGET = cands[0]
print(f"  target: {TARGET.name}")

hr("2. what the checkpoint carries")
ck = torch.load(TARGET, map_location="cpu", weights_only=False)
print(f"  top-level keys: {sorted(ck.keys())}")
print(f"  epoch          = {ck.get('epoch')!r}      <- resume continues at epoch+1")
print(f"  step           = {ck.get('step')!r}")
print(f"  vqvae_is_fit   = {ck.get('vqvae_is_fit')!r}")
print(f"  lr             = {ck.get('lr')!r}")
print(f"  policy_steps_effective = {ck.get('policy_steps_effective')!r}")

gate = {}

# The saved launch config, for cross-checking against the yaml on disk.
run_cfg = ck.get("config", {})
print(f"\n  ckpt['config'] (as launched): epochs={run_cfg.get('epochs')!r} "
      f"lr={run_cfg.get('lr')!r} weight_decay={run_cfg.get('weight_decay')!r} "
      f"betas={run_cfg.get('betas')!r} eval_freq_epochs={run_cfg.get('eval_freq_epochs')!r} "
      f"save_ckpt={run_cfg.get('save_ckpt')!r}")

# Gate 1 (resume_optimizer step 1): optimizer state must be present at all.
# GroupedOptimizer.state_dict() is keyed by optimizer index, so the Adam slots live
# one level down in opt[i]['state'] -- checking opt['state'] would always report 0.
opt = ck.get("optim")
gate["optim state present"] = opt is not None
print(f"\n  'optim' present = {opt is not None}", end="")
if opt is not None:
    sub = sorted(opt.keys(), key=lambda k: int(k))
    print(f" (keyed by optimizer index {sub})")
    for i in sub:
        s = opt[i]
        n_slots = len(s.get("state", {}))
        n_groups = len(s.get("param_groups", []))
        print(f"    opt{i}: {n_slots} Adam state slot(s) over {n_groups} param_group(s)")
    if sum(len(opt[i].get("state", {})) for i in sub) == 0:
        gate["optim state present"] = False
        print("    [FATAL] every sub-optimizer has an EMPTY state -- Adam momentum was "
              "not saved; resume_optimizer would not reject this, but the continuation "
              "would silently restart momentum")

# Gate 2 (resume_optimizer step 2): ordered param NAMES must exist and match the
# layout this process would build. We can only check presence + shape here; the
# match itself needs the live model, which step 4 of this probe builds.
names = ck.get("optim_param_names")
gate["optim_param_names present"] = names is not None
if names is not None:
    d = {int(k): v for k, v in names.items()}
    print(f"  optim_param_names: {len(d)} optimizer(s), "
          f"groups per optimizer {[len(v) for _, v in sorted(d.items())]}, "
          f"params per optimizer {[sum(len(g) for g in v) for _, v in sorted(d.items())]}")
    print(f"    opt0 group0[0:3] = {d[0][0][:3]}")

# Gate 3 (resume_optimizer step 4): the recorded LR must equal the configured LR.
gate["lr recorded"] = ck.get("lr") is not None
# Gate 4/5 (train.py:1014-1023 and :1036-1042).
tn = ck.get("trainable_param_names")
gate["trainable_param_names present"] = tn is not None
print(f"  trainable_param_names: {len(tn) if tn else None} entries"
      + (f", first={tn[0]!r}" if tn else ""))
# RNG is not a gate -- a legacy ckpt without it warns and continues -- but it decides
# whether the continuation is bit-reproducible.
gate["rng state present"] = ck.get("rng") is not None
print(f"  rng streams = {sorted(ck['rng'].keys()) if ck.get('rng') else None}")

hr("3. can the model that wrote this ckpt be rebuilt? (config + anchors)")
cfg_path = ROOT / "configs" / "pusht_dino_p256_td_k40.yaml"
print(f"  config {cfg_path.name}: {'exists' if cfg_path.is_file() else '[MISSING]'}")
if cfg_path.is_file():
    txt = cfg_path.read_text().replace("\r", "")
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith(("n_modes:", "anchor_artifact:", "action_head:", "epochs:",
                         "infer_steps:", "infer_t_start:", "t_trunc:")):
            print(f"    {s}")
    anchor = None
    for line in txt.splitlines():
        if line.strip().startswith("anchor_artifact:"):
            anchor = line.split(":", 1)[1].strip()
    if anchor:
        ap = ROOT / anchor
        print(f"  anchor file {anchor}: "
              + (f"exists ({ap.stat().st_size:,} B)" if ap.is_file() else "[MISSING]"))

# The decisive check: build the model exactly as the training run did, load the ckpt
# into it, and run the same name comparison resume_optimizer would run. Anything the
# static inspection above cannot see (a shape drift, a renamed module) shows up here.
hr("4. live rebuild: does this process's optimizer layout match the ckpt's?")
try:
    sys.path.insert(0, str(ROOT))
    from patch_policy.config import Config
    from patch_policy.train import (_optim_lrs, _optim_param_names,
                                    _trainable_param_names, build_model)
    cfg = Config.from_yaml(str(cfg_path))
    cfg.save_dir = str(RUN.parent)
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    print("  model.load_state_dict(ck['model']): OK (strict=True)")

    # Same call train.py:734 makes, so the layout below is the layout a real resume
    # would compare against -- not a hand-built approximation of it.
    optim = model.configure_optimizers(cfg.weight_decay, cfg.lr, tuple(cfg.betas))
    optims = getattr(optim, "optimizers", None) or (
        list(optim) if isinstance(optim, (list, tuple)) else [optim])
    print(f"  configure_optimizers(wd={cfg.weight_decay}, lr={cfg.lr}, "
          f"betas={tuple(cfg.betas)}) -> {len(optims)} optimizer(s)")

    cur_names = _optim_param_names(model, optim)
    saved_names = {int(k): v for k, v in names.items()} if names else None
    match = (saved_names == cur_names)
    gate["param layout matches (would pass resume_optimizer)"] = match
    print(f"  saved layout == current layout : {match}")
    if not match and saved_names is not None:
        for i in sorted(set(saved_names) | set(cur_names)):
            a, b = saved_names.get(i, []), cur_names.get(i, [])
            if a != b:
                print(f"    opt{i}: saved {sum(len(g) for g in a)} params vs "
                      f"current {sum(len(g) for g in b)}")

    cur_lr = _optim_lrs(optim)
    saved_lr = [float(x) for x in (ck["lr"] if isinstance(ck["lr"], list) else [ck["lr"]])]
    gate["lr matches"] = (saved_lr == cur_lr)
    print(f"  saved lr {saved_lr} == current lr {cur_lr} : {saved_lr == cur_lr}")

    cur_tn = _trainable_param_names(model)
    gate["trainable set matches"] = (sorted(tn) == cur_tn)
    print(f"  saved {len(tn)} trainable == current {len(cur_tn)} : {sorted(tn) == cur_tn}")
    gate["vqvae_is_fit consistent"] = (bool(ck.get("vqvae_is_fit")) is True)
    print(f"  vqvae_is_fit: ckpt={ck.get('vqvae_is_fit')} model={model.vqvae_is_fit}")
except Exception as e:  # noqa: BLE001 -- this probe's job is to report, not to raise
    import traceback
    traceback.print_exc()
    print(f"\n  [ERROR] live rebuild failed: {type(e).__name__}: {e}")
    gate["live rebuild"] = False

hr("5. metrics history (merged into the resumed run)")
mp = RUN / "metrics.json"
if mp.is_file():
    m = json.load(open(mp))
    met = m.get("metrics", {})
    tl = met.get("train_loss", {})
    vl = met.get("val_loss", {})
    print(f"  metrics.json: {len(tl)} train_loss entries, {len(vl)} val_loss entries")
    print(f"  train_loss epochs: {sorted(int(k) for k in tl)}")
    print(f"  val_loss epochs  : {sorted(int(k) for k in vl)}")
else:
    print("  [MISSING] metrics.json -- resume would start a fresh history file")

hr("6. how far a resume would run")
# train.py:1088 is `for epoch in range(resume_epoch, cfg.epochs)`, so the ceiling is
# cfg.epochs -- which --epochs sets. The K=40 yaml says 100 while the sweep ran 40,
# so a bare `--resume` inherits 100 from the config, not 40 from the ckpt.
resume_epoch = int(ck.get("epoch", 0))
print(f"  ckpt epoch      = {resume_epoch}  -> the loop restarts at epoch index {resume_epoch}")
print(f"  yaml cfg.epochs = 100")
print(f"  a bare --resume would run {100 - resume_epoch} more epochs (e={resume_epoch+1}..100)")
print(f"  to stop at 60: --epochs 60 -> {60 - resume_epoch} more epochs")
print(f"  next epoch SAVES land at e in ckpt_epochs, i.e. whatever --ckpt-epochs says;")
print(f"  e=41..50 are NOT saved unless 50 is in --ckpt-epochs, and e=50 does not")
print(f"  collide with the existing ep10/20/30/40 files.")

hr("VERDICT")
for k, v in gate.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print()
print("  RESUMABLE" if all(gate.values()) else "  NOT RESUMABLE AS-IS -- see FAILs above")
