"""Shape and compile-key contract for fused ReLU-plus-Conv2d CROWN tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

FUSED_CROWN_CONV2D_SCHEMA_VERSION = "boundflow.fused_crown_conv2d/v1"
COEFFICIENT_LAYOUT = "DSCOHW"
WEIGHT_LAYOUT = "OIHW"
OUTPUT_LAYOUT = "DSCIHW"


def _pair(value: Tuple[int, int], *, name: str) -> Tuple[int, int]:
    if len(value) != 2 or any(int(item) <= 0 for item in value):
        raise ValueError(f"{name} must contain two positive integers")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class FusedCrownConv2dSignature:  # pylint: disable=too-many-instance-attributes
    """Static Conv signature; weight values remain runtime inputs."""

    domain_batch: int
    spec_batch: int
    input_channels: int
    input_height: int
    input_width: int
    output_channels: int
    output_height: int
    output_width: int
    kernel_height: int
    kernel_width: int
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int] = (1, 1)
    groups: int = 1
    bias_present: bool = True
    coefficient_layout: str = COEFFICIENT_LAYOUT
    weight_layout: str = WEIGHT_LAYOUT
    output_layout: str = OUTPUT_LAYOUT
    dtype: str = "float32"
    target: str = "cuda"
    compute_capability: str = "sm_89"
    schedule_id: str = "output_gather_128t_v1"

    def validate(self) -> None:  # pylint: disable=too-many-branches
        """Reject every Conv attribute outside the frozen PR-12 v0/v1 subset."""

        for name in (
            "domain_batch",
            "spec_batch",
            "input_channels",
            "input_height",
            "input_width",
            "output_channels",
            "output_height",
            "output_width",
            "kernel_height",
            "kernel_width",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        stride_h, stride_w = _pair(self.stride, name="stride")
        dilation_h, dilation_w = _pair(self.dilation, name="dilation")
        if len(self.padding) != 2 or any(int(item) < 0 for item in self.padding):
            raise ValueError("padding must contain two non-negative integers")
        if self.groups != 1:
            raise NotImplementedError("fused Conv2d v1 only supports groups=1")
        if (dilation_h, dilation_w) != (1, 1):
            raise NotImplementedError("fused Conv2d v1 only supports dilation=1")
        if (stride_h, stride_w) not in {(1, 1), (2, 2)}:
            raise NotImplementedError("fused Conv2d v1 only supports stride=1 or 2")
        if (self.kernel_height, self.kernel_width) not in {(1, 1), (3, 3)}:
            raise NotImplementedError("fused Conv2d v1 only supports 1x1 or 3x3")
        if tuple(int(item) for item in self.padding) not in {(0, 0), (1, 1)}:
            raise NotImplementedError("fused Conv2d v1 only supports padding=0 or 1")
        if (
            self.coefficient_layout != COEFFICIENT_LAYOUT
            or self.weight_layout != WEIGHT_LAYOUT
            or self.output_layout != OUTPUT_LAYOUT
        ):
            raise NotImplementedError("unsupported fused Conv2d layout")
        if self.dtype != "float32":
            raise NotImplementedError("fused Conv2d v1 only supports float32")
        if not self.target.startswith("cuda"):
            raise NotImplementedError("fused Conv2d v1 only supports CUDA")
        if not self.compute_capability.startswith("sm_"):
            raise ValueError("compute_capability must use the sm_NN form")
        if self.schedule_id != "output_gather_128t_v1":
            raise NotImplementedError(f"unsupported schedule_id: {self.schedule_id}")
        expected_h, expected_w = self.expected_output_spatial()
        if (self.output_height, self.output_width) != (expected_h, expected_w):
            raise ValueError(
                "Conv2d output shape does not match the explicit input shape: "
                f"expected {(expected_h, expected_w)}, got "
                f"{(self.output_height, self.output_width)}"
            )
        self.output_padding()

    def expected_output_spatial(self) -> Tuple[int, int]:
        """Return the forward Conv2d output shape from the explicit input shape."""

        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation
        output_h = (
            self.input_height + 2 * pad_h - dilation_h * (self.kernel_height - 1) - 1
        ) // stride_h + 1
        output_w = (
            self.input_width + 2 * pad_w - dilation_w * (self.kernel_width - 1) - 1
        ) // stride_w + 1
        return int(output_h), int(output_w)

    def output_padding(self) -> Tuple[int, int]:
        """Derive ConvTranspose output padding from the original Primal input shape."""

        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation
        base_h = (
            (self.output_height - 1) * stride_h
            - 2 * pad_h
            + dilation_h * (self.kernel_height - 1)
            + 1
        )
        base_w = (
            (self.output_width - 1) * stride_w
            - 2 * pad_w
            + dilation_w * (self.kernel_width - 1)
            + 1
        )
        output_padding = self.input_height - base_h, self.input_width - base_w
        if not (
            0 <= output_padding[0] < stride_h and 0 <= output_padding[1] < stride_w
        ):
            raise ValueError(
                "explicit input shape cannot be recovered by ConvTranspose2d: "
                f"output_padding={output_padding}, stride={self.stride}"
            )
        return output_padding

    @property
    def target_string(self) -> str:
        """Return the CUDA target with the cache-keyed compute capability."""

        return f"{self.target} -arch={self.compute_capability}"


__all__ = [
    "COEFFICIENT_LAYOUT",
    "FUSED_CROWN_CONV2D_SCHEMA_VERSION",
    "FusedCrownConv2dSignature",
    "OUTPUT_LAYOUT",
    "WEIGHT_LAYOUT",
]
