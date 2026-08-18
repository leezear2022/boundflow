# Audit fsg4-b4b0-five-fresh-20260818/r001/audit

- round: 1
- delivery: fsg4-b4b0-five-fresh-20260818/r001/delivery
- verdict: request_changes
- from: external-model -> to: codex
- ts: 2026-08-18T05:37:12Z

## Findings

### F1 [major] scripts/run_fsg4_b4b_five_fresh_artifact.py:_discrete_projection/_summary

- evidence: coordinated local integrity cases that synchronously rewrite all 10 capture topology hashes or lineage source tensor hashes, then resign capture hashes, PT file digests and manifest hash, are accepted by root replay with exit 0
- advice: bind frozen per-anchor topology and lineage identities to protocol/source capture, add coordinated-all-runs negative tests, regenerate clean-source artifact, and submit Round 2

## Summary

reject: AC1-AC3 and AC5-AC6 pass, but AC4 fails because cross-run equality rejects only single-capture identity drift and accepts coordinated topology/lineage identity rewrites; B4-B0 cannot close and B4-B1/B4-B2/performance remain closed. Full evidence: r001/audit_report_full.md
