from typing import Union, Optional, List
import onnx
from ...ir.primal import BFPrimalProgram, Source, BFPrimalGraph

def import_onnx(
    model_or_path: Union[str, onnx.ModelProto],
    *,
    do_shape_infer: bool = True,
    input_shapes: Optional[List[List[int]]] = None,
    normalize: bool = True
) -> BFPrimalProgram:
    """
    Import an ONNX model into BoundFlow Primal IR.
    
    Args:
        model_or_path: Path to .onnx file or loaded ModelProto.
        do_shape_infer: Whether to run onnx.shape_inference.
        input_shapes: Optional manual shape override.
        normalize: Whether to run normalization to primitive ops.
    """
    # TODO: Load ONNX and convert
    # if isinstance(model_or_path, str):
    #     model = onnx.load(model_or_path)
    
    # Placeholder return
    return BFPrimalProgram(
        source=Source.ONNX,
        graph=BFPrimalGraph(),
        params={},
        tensor_meta={}
    )
