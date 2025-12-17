from abc import ABC, abstractmethod
from typing import Any, List, Dict
from ..ir.bound import DomainState

class AbstractDomain(ABC):
    """
    Interface for Abstract Domains (e.g. Interval, DeepPoly, Zonotope).
    Each domain must implement a set of transformers for the primitive ops.
    """
    
    @property
    @abstractmethod
    def domain_id(self) -> str:
        """Unique identifier for the domain."""
        pass

    @abstractmethod
    def affine_transformer(self, 
                           state_in: DomainState, 
                           weight: Any, 
                           bias: Any, 
                           **attrs) -> DomainState:
        """Transformer for Affine operations (Linear/Conv)."""
        pass

    @abstractmethod
    def relu_transformer(self, state_in: DomainState) -> DomainState:
        """Transformer for ReLU activation."""
        pass
    
    @abstractmethod
    def elementwise_transformer(self, 
                                states_in: List[DomainState], 
                                op: str) -> DomainState:
        """Transformer for elementwise ops (Add, Mul)."""
        pass

    # Add other primitives (Pool, Reshape) as needed
