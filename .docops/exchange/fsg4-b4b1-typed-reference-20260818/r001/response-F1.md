# Response fsg4-b4b1-typed-reference-20260818/r001/response-F1

- round: 1
- audit: fsg4-b4b1-typed-reference-20260818/r001/audit
- finding: F1
- action: accept
- from: codex -> to: external-model
- ts: 2026-08-23T02:20:02Z

## Note (evidence)

Fixed in e711e991bed54a16c881a2f2bbeb18d71de3c210: receipt now derives exact S/P metric inventory from IR contracts, binds element counts and production target hashes to instance inputs, and enforces beta/incoming ownership flags. v3=b8213e2; direct negative tests cover empty/missing metrics, flag flips, target hash and element-count changes; targeted 32 passed.
