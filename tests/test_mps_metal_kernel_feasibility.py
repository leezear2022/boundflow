from scripts.report_mps_metal_kernel_feasibility import _geomean, _parse_sizes


def test_parse_sizes() -> None:
    assert _parse_sizes("4, 8,16") == [4, 8, 16]


def test_geomean() -> None:
    assert round(_geomean([1.0, 4.0]) or 0.0, 6) == 2.0
