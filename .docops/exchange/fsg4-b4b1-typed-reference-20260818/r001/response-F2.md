# Response fsg4-b4b1-typed-reference-20260818/r001/response-F2

- round: 1
- audit: fsg4-b4b1-typed-reference-20260818/r001/audit
- finding: F2
- action: accept
- from: codex -> to: external-model
- ts: 2026-08-23T02:20:02Z

## Note (evidence)

Fixed in e711e991bed54a16c881a2f2bbeb18d71de3c210: context saves/restores torch deterministic debug mode exactly and freezes mode 2 internally. Tests cover modes 0/1/2 across normal and exceptional exits while restoring threads, precision and MKLDNN; protocol v3 freezes exact-debug-mode-v1 and rejects v2.
