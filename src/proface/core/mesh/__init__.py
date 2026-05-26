# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

__all__ = [
    "DIM",
    "Elements",
    "LoadCase",
    "Mesh",
    "NamedRegion",
    "Nodes",
    "Quantity",
    "QuantityKey",
    "Set",
    "Topology",
    "__version__",
]

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

from ._constants import DIM
from .container import Elements, Mesh, NamedRegion, Nodes, Set, Topology
from .results import LoadCase, Quantity, QuantityKey
