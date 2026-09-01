# Root CROWN terminal TIR five-pair artifact

This artifact contains a local-path-free projection of ten fresh same-solver processes. It binds the final semantics, wrapper timing, root timing, compiled-module identities, and activation receipts.

- query geomean: `1.007266894197x`
- root geomean: `1.016167981494x`
- optimizer-transaction geomean: `1.024849133115x`
- maximum lower difference: `2.26497650146e-06`
- decision: mechanism correct; no stable query speedup claim

Replay with:

```bash
python scripts/package_root_crown_terminal_five_pair.py replay --artifact artifacts/root-crown-terminal-tir/resnet2b-prop0-v1
```
