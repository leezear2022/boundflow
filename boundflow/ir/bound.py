from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .primal import BFPrimalProgram

@dataclass
class DomainState:
    """Abstract base class for domain states (e.g., Interval (l,u), Linear (A,b))."""
    pass

@dataclass
class Spec:
    """Verification specification (perturbation, objectives)."""
    perturbation: Dict[str, Any] # e.g. {"x": ("L_inf", 0.01)}
    objectives: Any # e.g. target labels

@dataclass
class ApplyTransformer:
    """A node in the Bound Graph representing a domain transformation."""
    op_type: str
    name: str # Unique ID
    inputs: List[str] # Input state names
    outputs: List[str] # Output state names
    attrs: Dict[str, Any]
    
    # Original primal node reference (optional)
    primal_node_name: Optional[str] = None

@dataclass
class BFBoundGraph:
    nodes: List[ApplyTransformer]
    inputs: List[str]
    outputs: List[str]

@dataclass
class BFBoundProgram:
    primal: BFPrimalProgram
    domain_id: str # e.g. "interval", "deepoly"
    spec: Spec
    bound_graph: BFBoundGraph
    
    # Maps state_name -> DomainState object (if computed/cached)
    state_table: Dict[str, DomainState] = field(default_factory=dict)
