# Delivery fsg4-b3-formal-timing-20260814/r002/delivery

- round: 2
- from: codex -> to: external-model
- base commit: 36e9069ca4f21183c9b36d74024de0ca8b20f59c
- result commit: d1c95059bb399b7cb01ce6a8b97f5149e21ae6de
- ts: 2026-08-14T14:02:22Z

## Changed files

- artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/
- artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1-tamper-report.json
- tests/test_fsg4_b3_same_solver_artifact.py
- gemini_doc/change_2026-08-14_fsg4_b3_formal_timing_closure.md
- gemini_doc/fsg4_b3_formal_timing_external_audit_handoff_2026_08_14.md

## Claims

- Corrected immutable round after executor-only metadata preflight. 36/36 fresh same-solver workers passed correctness/environment/measurement/activation; B2/B3 core 1.071617x and query 1.006623x, B0/B3 query 0.910001x, therefore VALIDATED-REDUCED-B3 only

## Validation

- `python scripts/run_fsg4_b3_same_solver_experiment.py replay --artifact-dir artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1` -> pass
- `python scripts/probe_fsg4_b3_same_solver_artifact_tamper.py --artifact-dir artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1 --report /tmp/fsg4-b3-audit-tamper.json` -> pass
- `pytest -q tests/test_fsg3_same_solver*.py tests/test_fsg4_b3*.py` -> pass
- `pytest -q tests` -> pass

## Known limitations

- single RTX 4060 Laptop GPU, ResNet2B property 0, fixed one-iteration solver prefix; artifact performance_claimed=false

## Risks

- B3 remains about 9.89 percent slower than B0 query; no memory win; raw solver logs retain manifest-bound upstream trailing whitespace

## Open questions

- Independently audit round 2 AC1-AC7; approve VALIDATED-REDUCED-B3 and permit only B4 cumulative candidate?
