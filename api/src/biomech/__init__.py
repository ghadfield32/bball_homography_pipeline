# api/src/biomech/__init__.py
"""
Biomechanics analysis package.

Provides tools for working with motion capture and pose estimation data,
aligned with SPL-Open-Data standards.
"""
from api.src.biomech.spl_adapter import (
    SPLAdapter,
    SPLTransform,
    create_spl_adapter,
)

__all__ = [
    "SPLAdapter",
    "SPLTransform",
    "create_spl_adapter",
]
