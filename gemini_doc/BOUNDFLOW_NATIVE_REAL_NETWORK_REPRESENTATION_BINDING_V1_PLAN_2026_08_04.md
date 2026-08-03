---
status: complete
updated: 2026-08-03T20:49:48Z
type: plan
topic: boundflow
slug: BOUNDFLOW_NATIVE_REAL_NETWORK_REPRESENTATION_BINDING_V1
stage: s01
---

# Native Real-Network Representation Binding v1 Plan

## Goal

- Close the semantic gap between Plan/Schedule representation decisions and the
  Bound/Task/Launch program that actually executes for native plain-CROWN.
- Produce replay-grade evidence that a selected structured-affine policy creates
  explicit cast/materialize Bound operations and preserves the dense result.

## Scope

- Add a deterministic two-policy native Plan variant: all-dense and
  structured-affine-with-dense-boundaries.
- Bind one exact source PlanInstance and ScheduleModule to either the unchanged
  dense Bound module or a distinct rewritten execution Bound module.
- Record one-to-one linkage from every selected transition candidate and source
  Schedule materialization action to an inserted execution Bound operation.
- Lower and execute a fresh Plan/Task/Schedule stack for the bound execution
  module; do not claim that a source PlanTemplate is valid for a changed Bound
  hash.
- Generate and semantically replay a frozen real-network artifact when the
  pinned ResNet source is locally available.
- Exclude arbitrary mixed per-region policies, compressed physical storage,
  CUDA performance, allocator, OOM, and Pareto claims from v1.

## Tasks

1. Freeze policy IDs, transition signatures, binding-trace schema, hashes, and
   fail-closed validation rules.
2. Build globally compatible dense and structured-affine Plan choices over the
   dense source Bound graph.
3. Implement the compile-time binder and reject policy, Schedule action, Bound
   rewrite, or hash mismatches.
4. Generalize the reference execution template to admit explicit structured
   values while recording their physical size as dense-equivalent.
5. Prove both selected policies lower through Bound -> Plan -> Task -> Schedule
   -> Launch and return numerically equivalent lower bounds.
6. Freeze raw/summary/manifest evidence and a replay that independently rebuilds
   the semantic binding rather than trusting stored summary fields.

## Validation

- Focused positive tests for dense and structured policy selection, exact
  transition/action/op linkage, launch ownership, and numerical equivalence.
- Negative tests for altered source hashes, policy decisions, transition sets,
  Schedule actions, execution Bound transitions, and stored replay payloads.
- Real ResNet CPU execution and artifact replay when the pinned ONNX/VNNLIB
  inputs are present; otherwise record a fail-closed source-unavailable boundary
  without upgrading the real-network claim.
- `black`, `mypy`, `pylint`, focused pytest, full `pytest tests`,
  `git diff --check`, and `dol lint --soft`.

Closure result: focused representation/Plan/artifact tests `25 passed`; full
suite `496 passed, 37 skipped`; Black clean, Mypy clean, Pylint `10.00/10`,
`git diff --check` clean; real ResNet generate and semantic replay exit 0.

## Rollback

- The new compiler path is additive. Roll back its module, tests, runner,
  artifact, and authority-document entries; the merged NRIR-1/2/3 dense native
  path remains unchanged.

## Links

- changelog: `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_REPRESENTATION_BINDING_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
