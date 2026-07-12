from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tir as T


@T.prim_func
def add_one(a: T.Buffer((256,), "float32"), b: T.Buffer((256,), "float32")):
    T.func_attr({"tir.noalias": True})
    for block in T.thread_binding(4, thread="blockIdx.x"):
        for thread in T.thread_binding(64, thread="threadIdx.x"):
            index = block * 64 + thread
            b[index] = a[index] + T.float32(1)


def main() -> None:
    module = tvm.build(add_one, target="cuda")
    device = tvm.cuda(0)
    source = np.arange(256, dtype="float32")
    a = tvm.runtime.tensor(source, device)
    b = tvm.runtime.tensor(np.zeros_like(source), device)
    module(a, b)
    np.testing.assert_allclose(b.numpy(), source + 1)
    print("TVM TIR CUDA smoke: OK")


if __name__ == "__main__":
    main()
