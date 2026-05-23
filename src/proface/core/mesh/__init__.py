# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

__all__ = [
    "DIM",
    "Elements",
    "Mesh",
    "NamedRegion",
    "Nodes",
    "Set",
    "Topology",
    "__version__",
]

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

from .container import DIM, Elements, Mesh, NamedRegion, Nodes, Set, Topology
