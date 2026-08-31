#!/usr/bin/env python3
"""External-audit independent recompute for ASPLOS'27 S4-1B0 (AC3/AC4).

Stdlib-only, written by the external auditor. Parses the .bin sidecars
directly and recomputes the IEEE classifier, midpoint select and all counts
from raw bytes. Does not import boundflow/torch/tvm or the replay tool.
"""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

NUMEL = 18432
QNAN = 0x7FC00000
EXPMASK = 0x7F800000

failures = []


def check(cond, label):
    if not cond:
        failures.append(label)
        print("FAIL:", label)


def f32_from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def classify_bits(bits):
    """My own independent reading of the frozen policy:
    nonfinite iff all exponent bits set -> -128; +-0 -> 0; sign otherwise."""
    raw = bits & 0xFFFFFFFF
    if raw & EXPMASK == EXPMASK:
        return -128
    if raw & 0x7FFFFFFF == 0:
        return 0
    return -1 if raw & 0x80000000 else 1


def select_bits(selector, lower_bits, upper_bits):
    if selector == 1:
        return lower_bits & 0xFFFFFFFF
    if selector == -1:
        return upper_bits & 0xFFFFFFFF
    if selector == 0:
        lower = f32_from_bits(lower_bits)
        upper = f32_from_bits(upper_bits)
        # f32(f32(lower + upper) * f32(0.5)): two rounding steps
        summed = f32_from_bits(struct.unpack("<I", struct.pack("<f", lower + upper))[0])
        mid = f32_from_bits(struct.unpack("<I", struct.pack("<f", summed * 0.5))[0])
        return struct.unpack("<I", struct.pack("<f", mid))[0]
    return QNAN


def audit(root_str):
    root = Path(root_str)
    print(f"=== {root} ===")
    rows = [json.loads(line) for line in
            open(root / "raw/workers.jsonl", encoding="utf-8")]
    check(len(rows) == 11, "11 rows")
    seq = [r["worker_name"] for r in rows]
    expect_seq = ([f"positive-{i:02d}" for i in range(5)] + ["cache-00"]
                  + ["fault-classifier-policy", "fault-cache-source",
                     "fault-descriptor-dlpack", "fault-stream-launch",
                     "fault-invalid-selector-claim"])
    check(seq == expect_seq, "worker sequence")
    check(len({r["pid"] for r in rows}) == 11, "11 distinct pids")

    sidecar_shas = []
    for i in range(5):
        row = rows[i]
        blob = open(root / "raw/binary" / f"positive-{i:02d}.bin", "rb").read()
        check(len(blob) == 313344, f"pos{i} byte count")
        sha = hashlib.sha256(blob).hexdigest()
        sidecar_shas.append(sha)
        check(row["binary"]["sha256"] == sha, f"pos{i} sidecar hash binding")
        # fixed layout: coefficient | lower | upper (18432 f32 each),
        # selector (18432 i8), selected (18432 f32)
        coeff = struct.unpack("<18432I", blob[0:73728])
        lower = struct.unpack("<18432I", blob[73728:147456])
        upper = struct.unpack("<18432I", blob[147456:221184])
        selector = struct.unpack("<18432b", blob[221184:239616])
        selected = struct.unpack("<18432I", blob[239616:313344])
        counts = {"positive": 0, "negative": 0, "zero": 0, "invalid": 0}
        exact = 0
        mismatches = []
        for j in range(NUMEL):
            want_sel = classify_bits(coeff[j])
            if selector[j] != want_sel:
                mismatches.append(("selector", j, selector[j], want_sel))
                continue
            label = ("positive" if selector[j] == 1 else
                     "negative" if selector[j] == -1 else
                     "zero" if selector[j] == 0 else "invalid")
            counts[label] += 1
            want_bits = select_bits(selector[j], lower[j], upper[j])
            if selected[j] == want_bits:
                exact += 1
            else:
                mismatches.append(("selected", j, selected[j], want_bits))
        check(not mismatches, f"pos{i} mismatches {mismatches[:3]}")
        check(counts == {"positive": 8689, "negative": 9137,
                         "zero": 606, "invalid": 0}, f"pos{i} counts {counts}")
        check(exact == NUMEL, f"pos{i} bitwise exact {exact}/{NUMEL}")
        check(row["counts"] == counts, f"pos{i} row counts binding")
        # old sign-only binary policy would misclassify the zeros
        check(row["old_binary_zero_misclassified"] == 606,
              f"pos{i} old-policy counter")
        # descriptor/module evidence
        check(len(row["descriptor_hashes"]) == 5
              and row["dlpack_pointer_exact"] == 5,
              f"pos{i} descriptor/dlpack evidence")
        rec = row["module_receipt"]
        check(hashlib.sha256(json.dumps(
            rec, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest() == row["module_receipt_hash"],
            f"pos{i} module receipt hash")
        check(rec["performance_claimed"] is False, f"pos{i} receipt flag")
    check(len(set(sidecar_shas)) == 1, "5 sidecars byte-identical")
    print(f"  5 sidecars identical: {sidecar_shas[0][:16]}...")

    # AC4 cache row
    cache = rows[5]
    check(cache["events"] == ["miss", "hit"], "cache events")
    check((cache["compile_count"], cache["miss_count"], cache["hit_count"],
           cache["entry_count"]) == (1, 1, 1, 1), "cache counters")
    check(cache["same_module_receipt"] is True
          and cache["tensor_retention_count"] == 0, "cache receipt/retention")

    # AC4 fault rows
    expect_reasons = [
        "TERNARY_ENDPOINT_MIDPOINT_POLICY_MISMATCH",
        "TERNARY_ENDPOINT_DEVICE_SOURCE_MISMATCH",
        "TERNARY_ENDPOINT_DLPACK_IDENTITY_MISMATCH",
        "TERNARY_ENDPOINT_STREAM_IDENTITY_MISMATCH",
        "TERNARY_ENDPOINT_INVALID_SELECTOR_NOT_POISONED",
    ]
    for k, row in enumerate(rows[6:]):
        res = row["result"]
        check(res["reason"] == expect_reasons[k], f"fault{k} reason")
        check(res["context_is_none"] is True, f"fault{k} context none")
        for key, want in (("fallback_count", 0), ("retry_count", 0),
                          ("native_shadow_count", 0), ("eager_count", 0)):
            if key in res:
                check(res[key] == want, f"fault{k} {key}")
        if "launch_count" in res:
            check(res["launch_count"] == 0, f"fault{k} reject-before-launch")
    print("  cache miss/hit/compile=1/1/1; 5 fault reasons in frozen order")

    # summary binding
    summary = json.load(open(root / "summary.json", encoding="utf-8"))
    check(summary["positive_sidecar_sha256"] == sidecar_shas[0],
          "summary sidecar binding")
    check(summary["selector_counts"] ==
          {"positive": 8689, "negative": 9137, "zero": 606, "invalid": 0},
          "summary counts")
    check(summary["selected_bitwise_exact"] is True, "summary bitwise flag")
    check(summary["performance_claimed"] is False
          and summary["timing_recorded"] is False, "summary flags")
    check(summary["status"] ==
          "FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1B0",
          "summary status")


def main():
    roots = sys.argv[1:] or ["artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1"]
    for root in roots:
        audit(root)
    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURES")
        sys.exit(1)
    print("RESULT: S4-1B0 INDEPENDENT RECOMPUTE PASSED")


if __name__ == "__main__":
    main()
