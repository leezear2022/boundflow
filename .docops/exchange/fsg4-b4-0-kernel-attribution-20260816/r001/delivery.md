# Delivery fsg4-b4-0-kernel-attribution-20260816/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: 9adde46
- result commit: dcd128c
- ts: 2026-08-15T17:15:54Z

## Changed files

- boundflow/runtime/fsg4_b4_kernel_attribution.py
- scripts/run_fsg4_b4_kernel_attribution.py
- tests/test_fsg4_b4_kernel_attribution.py
- tests/test_fsg4_b4_kernel_attribution_runner.py
- artifacts/fsg4-b4-kernel-attribution/resnet2b-prop0-v1
- gemini_doc/change_2026-08-16_fsg4_b4_0_kernel_attribution_closure.md

## Claims

- Internal B4-0 attribution/opportunity only; no speedup or memory claim

## Validation

- `targeted=pass,related=pass,full=pass,replay=pass,tamper=pass,static` -> pass

## Known limitations

- Profiler timing is attribution-only; allocation delta is cumulative; external audit pending

## Risks

- 2307 kernels use explicit temporal-marker fallback; audit must distinguish from correlation evidence

## Open questions

- Approve VALIDATED-B4-0-OPPORTUNITY and open only B4-A?
