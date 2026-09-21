"""Patch Policy on PushT: RGB -> frozen DINOv2 -> Patch Policy Transformer -> action head.

The base is Patch Policy (upstream `Patch policy/upstream/`; the PushT port of it that this
was forked from is `sibling-project/`). The package is named `patch_policy` because
that IS the base — only the action head varies between arms (VQ-BeT vs truncated
diffusion), so naming it after either arm would misdescribe the other.

The `wemm` / `qwen3vl` encoder branches are inherited and DORMANT here: this project
runs `encoder_name: dino` (DINOv2-S/14, P=256, frozen) with `fusion.enabled: false`,
so neither conditioning stream is built. They are kept because the harness is shared
byte-for-byte with the arms that do use them, not because this arm does.
"""

__version__ = "0.1.0"
