# Delivery nrir45-20260805/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: b6eb6974c58289585dfa07a4daa7e7383c9bd7df
- result commit: af1031eee0740b68d258c5780aa898c30b2b6fe2
- ts: 2026-08-05T02:35:05Z

## Changed files

- boundflow/ir/prepared_intermediate_refinement.py
- boundflow/runtime/native_prepared_intermediate_refinement.py
- boundflow/runtime/native_prepared_per_child_refinement.py
- boundflow/runtime/native_prepared_shared_parametric_ancestral.py
- boundflow/runtime/native_prepared_objective_branch_shared_production_queue.py
- boundflow/runtime/native_prepared_root_projection_multi_clause_anytime.py
- scripts/run_prepared_intermediate_refinement_formal.py
- scripts/run_prepared_intermediate_refinement_global_formal.py
- tests/test_native_prepared_intermediate_refinement.py
- tests/test_native_objective_branch_scorer_ownership.py
- artifacts/prepared-intermediate-refinement

## Claims

- NRIR45 reduces repeated target-selection/full-validation/full-hash ownership cost while preserving exact fixed ResNet2B property0 CPU8 semantics; Phase A ratios are 0.727519/0.736603 and Phase B trace/measured ratios are 0.710268/0.615738; final remains 9/9 unknown and performance_claimed=false

## Validation

- `targeted tests` -> pass
- `full pytest 984 passed 37 skipped` -> pass
- `Phase A and Phase B replay plus tamper` -> pass
- `Black mypy Pylint` -> pass
- `DocOps validate and lint` -> pass

## Known limitations

- One fixed ResNet2B property0 CPU8 internal workload only; no fair competitor, GPU, multi-workload, property closure, 10x, or ASPLOS-ready claim

## Risks

- Fast-path safety depends on exact owner/container identity/Tensor version; auditor must probe mutation and full replay independently

## Open questions

- Do AC1-AC6 independently justify approve and allow executor to close the exchange and merge draft PR #56?
