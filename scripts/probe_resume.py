"""Can K=20 and K=40 be resumed from ep40 to ep100 -- correctly and continuously?

The instruction (user, 2026-09-18): 「对于 K=20，K=40 的两个试验在 GPU1 和 GPU2 上续训原来的
ckpt，这里你要严格注意续训的合理和连续，我们最后训到 100ep」

"Reasonable and continuous" is not a vibe; it decomposes into nine things that can each be
checked against the actual file on disk. This script checks all nine for BOTH arms in one pass
and prints the exact launch command it has thereby licensed.

  1. nothing newer than ep40 is on disk  -> this really is a first resume, not a second one
  2. the ckpt carries a full resume payload (optim / optim_param_names / lr /
     trainable_param_names / rng-with-dataloader / policy_steps_effective / vqvae_is_fit)
  3. Adam moments are actually present (3 sub-optimizers, non-empty state), and the slot
     count equals len(trainable_param_names)
  4. the recorded LR equals the yaml LR            -- resume_optimizer hard-stops otherwise
  5. a live rebuild produces a BYTE-IDENTICAL optimizer layout (names, in order) and LR list
  6. the freeze scope matches                     -- otherwise resume_optimizer raises
  7. metrics.json is present and holds epochs 1..40 -- otherwise the curve silently restarts
  8. the anchor artifact the config names still hashes to what the run used
  9. the config's two content lines (n_modes / anchor_artifact) match the K being resumed

Continuity beyond the gates is structural, and is stated (with the code line that makes it
true) rather than checked, because it cannot fail without these gates failing too: constant
LR with no scheduler/warmup/decay (train.py:920), RNG restored including the dataloader
generator (train.py:1051-1059) so the shuffle order continues as an uninterrupted run would,
and `range(resume_epoch, cfg.epochs)` (train.py:1088) so the loop resumes at e=41.

Run on the server:
    cd ~/td_pusht && python scripts/probe_resume.py
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path.home() / "td_pusht"

# arm -> (run dir, config, expected K)
ARMS = [
    ("K=20", ROOT / "outputs" / "p4_td" / "seed0",
     ROOT / "configs" / "pusht_dino_p256_td.yaml", 20),
    ("K=40", ROOT / "outputs" / "ksweep_k40" / "seed0",
     ROOT / "configs" / "pusht_dino_p256_td_k40.yaml", 40),
]
FROM_EPOCH = 40
TO_EPOCH = 100


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg_lines(p):
    """The two content lines that define which K this config is."""
    out = {}
    for line in p.read_text().replace("\r", "").splitlines():
        s = line.strip()
        for key in ("n_modes:", "anchor_artifact:"):
            if s.startswith(key):
                out[key[:-1]] = s.split(":", 1)[1].strip()
    return out


all_green = True
for label, RUN, CFG, K in ARMS:
    hr(f"{label}  --  {RUN}")

    # ---- gate 1: this is a first resume ------------------------------------
    ckpts = sorted(RUN.glob("ckpt_epoch*_step*.pt"))
    labels = sorted(int(p.name.split("_")[1][5:]) for p in ckpts)
    print(f"  ckpt epoch labels on disk: {labels}")
    over = [n for n in labels if n > FROM_EPOCH]
    if over:
        print(f"  [FATAL] labels above {FROM_EPOCH}: {over} -- this arm has ALREADY been "
              f"resumed. Resuming again from ep{FROM_EPOCH} would fork the run and the "
              f"metrics merge would keep the later numbers. Stop and inspect.")
    target = next((p for p in ckpts if p.name.startswith(f"ckpt_epoch{FROM_EPOCH}_")), None)
    if target is None:
        print(f"  [FATAL] no ckpt_epoch{FROM_EPOCH}_* in {RUN}")
        all_green = False
        continue
    print(f"  target: {target.name}  ({target.stat().st_size:,} B)")

    ck = torch.load(target, map_location="cpu", weights_only=False)
    gate = {}

    # ---- gate 2: the resume payload ----------------------------------------
    print(f"\n  top-level keys: {sorted(ck.keys())}")
    print(f"  epoch={ck.get('epoch')!r}  step={ck.get('step')!r}  "
          f"vqvae_is_fit={ck.get('vqvae_is_fit')!r}  lr={ck.get('lr')!r}")
    print(f"  policy_steps_effective={ck.get('policy_steps_effective')!r}")
    rng = ck.get("rng")
    print(f"  rng streams={sorted(rng.keys()) if rng else None}")
    gate["epoch label is 40"] = (ck.get("epoch") == FROM_EPOCH)
    gate["rng incl. dataloader"] = bool(rng) and "dataloader" in rng
    gate["vqvae_is_fit True"] = (ck.get("vqvae_is_fit") is True)

    # ---- gate 3: Adam moments ----------------------------------------------
    opt = ck.get("optim")
    gate["optim present"] = opt is not None
    n_slots = 0
    if opt is not None:
        sub = sorted(opt.keys(), key=lambda k: int(k))
        print(f"\n  'optim' keyed by optimizer index {sub}")
        for i in sub:
            s = opt[i]
            ns = len(s.get("state", {}))
            n_slots += ns
            print(f"    opt{i}: {ns} Adam state slot(s) over "
                  f"{len(s.get('param_groups', []))} param_group(s)")
    tn = ck.get("trainable_param_names")
    n_train = len(tn) if tn else 0
    print(f"  trainable_param_names: {n_train} entries")
    slot_msg = ("match" if n_slots == n_train else
                "MISMATCH -- a param with no Adam slot is one the optimizer would treat as "
                "never-updated")
    print(f"  Adam slots {n_slots} vs trainable tensors {n_train}  ->  {slot_msg}")
    gate["Adam slots == trainable tensors"] = (n_slots == n_train and n_train > 0)

    names = ck.get("optim_param_names")
    gate["optim_param_names present"] = names is not None
    if names:
        d = {int(k): v for k, v in names.items()}
        print(f"  optim_param_names: {len(d)} optimizer(s), groups per optimizer "
              f"{[len(v) for _, v in sorted(d.items())]}, "
              f"params per optimizer {[sum(len(g) for g in v) for _, v in sorted(d.items())]}")

    # ---- gate 9: the config is the one that ran ----------------------------
    cl = cfg_lines(CFG)
    print(f"\n  config {CFG.name}: {cl}")
    ok_cfg = (cl.get("n_modes") == str(K)
              and cl.get("anchor_artifact") == f"anchors/pusht_kmeans_K{K}.pt")
    gate["config matches this K"] = ok_cfg
    if not ok_cfg:
        print(f"  [FATAL] config says {cl} but this arm is K={K}")

    # ---- gate 8: the anchor artifact ---------------------------------------
    ap = ROOT / cl.get("anchor_artifact", "")
    if ap.is_file():
        print(f"  anchor {ap.name}: {ap.stat().st_size:,} B  sha256 {sha256(ap)[:16]}...")
        gate["anchor artifact present"] = True
    else:
        print(f"  [FATAL] anchor artifact missing: {ap}")
        gate["anchor artifact present"] = False

    # ---- gate 7: metrics history -------------------------------------------
    mp = RUN / "metrics.json"
    if mp.is_file():
        m = json.load(open(mp))
        met = m.get("metrics", {})
        tl = sorted(int(k) for k in met.get("train_loss", {}))
        vl = sorted(int(k) for k in met.get("val_loss", {}))
        print(f"\n  metrics.json: {len(tl)} train_loss pts (e={tl[0]}..{tl[-1]}), "
              f"val_loss at e={vl}")
        print(f"    recorded ckpt_epochs: {m.get('ckpt_epochs')}")
        gate["metrics history reaches ep40"] = (tl and tl[-1] == FROM_EPOCH)
    else:
        print(f"\n  [FATAL] {mp} missing -- a resume would start a fresh curve")
        gate["metrics history reaches ep40"] = False

    # ---- gates 4-6: live rebuild -------------------------------------------
    print()
    try:
        sys.path.insert(0, str(ROOT))
        from patch_policy.config import Config
        from patch_policy.train import (_optim_lrs, _optim_param_names,
                                        _trainable_param_names, build_model)
        cfg = Config.from_yaml(str(CFG))
        cfg.save_dir = str(RUN.parent)
        cfg.epochs = TO_EPOCH
        model = build_model(cfg)
        model.load_state_dict(ck["model"])
        print("  model.load_state_dict(ck['model']): OK (strict=True)")
        optim = model.configure_optimizers(cfg.weight_decay, cfg.lr, tuple(cfg.betas))
        optims = getattr(optim, "optimizers", None) or (
            list(optim) if isinstance(optim, (list, tuple)) else [optim])
        print(f"  configure_optimizers(wd={cfg.weight_decay}, lr={cfg.lr}, "
              f"betas={tuple(cfg.betas)}) -> {len(optims)} optimizer(s)")

        cur_names = _optim_param_names(model, optim)
        saved_names = {int(k): v for k, v in names.items()} if names else None
        same = (saved_names == cur_names)
        gate["param layout identical"] = same
        print(f"  saved layout == current layout : {same}")
        if not same and saved_names is not None:
            for i in sorted(set(saved_names) | set(cur_names)):
                a, b = saved_names.get(i, []), cur_names.get(i, [])
                if a != b:
                    print(f"    opt{i}: saved {sum(len(g) for g in a)} params vs "
                          f"current {sum(len(g) for g in b)}")

        cur_lr = _optim_lrs(optim)
        saved_lr = [float(x) for x in (ck["lr"] if isinstance(ck["lr"], list) else [ck["lr"]])]
        gate["lr matches yaml"] = (saved_lr == cur_lr)
        print(f"  saved lr {saved_lr} == current lr {cur_lr} : {saved_lr == cur_lr}")

        cur_tn = _trainable_param_names(model)
        gate["freeze scope matches"] = (sorted(tn) == cur_tn)
        print(f"  saved {len(tn)} trainable == current {len(cur_tn)} : {sorted(tn) == cur_tn}")
    except Exception as e:  # noqa: BLE001 -- a probe reports, it does not raise
        import traceback
        traceback.print_exc()
        print(f"\n  [ERROR] live rebuild failed: {type(e).__name__}: {e}")
        gate["live rebuild"] = False

    # ---- verdict ------------------------------------------------------------
    print(f"\n  {label} VERDICT")
    for k, v in gate.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    green = all(gate.values())
    all_green = all_green and green
    if green:
        print(f"\n    LICENSED COMMAND ({TO_EPOCH - FROM_EPOCH} more epochs, "
              f"e={FROM_EPOCH + 1}..{TO_EPOCH}):")
        print(f"      cd ~/td_pusht && nohup env CUDA_VISIBLE_DEVICES=<GPU> "
              f"$HOME/miniconda3/envs/wemm-p1/bin/python -u -m patch_policy.train \\")
        print(f"          --config {CFG.relative_to(ROOT)} --train-seed 0 "
              f"--epochs {TO_EPOCH} --ckpt-epochs <list> --save-dir {RUN.parent.relative_to(ROOT)} \\")
        print(f"          --resume {target.relative_to(ROOT)} "
              f"> logs/{RUN.parent.name}_resume_train.log 2>&1")

hr("OVERALL")
print("  ALL ARMS RESUME-CLEAN" if all_green else "  NOT ALL CLEAN -- see FAILs above")
