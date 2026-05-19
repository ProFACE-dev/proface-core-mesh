# SPDX-FileCopyrightText: 2026 Stefano Miccoli <stefano.miccoli@polimi.it>
# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto

import attrs
import numpy as np
import numpy.typing as npt

from . import Topology

RES_DTYPE = np.float32


class ResultsKey(Enum):
    S = auto()
    SP = auto()
    COORD = auto()
    IVOL = auto()


@attrs.frozen
class ElementIP:
    topology: Topology
    values: npt.NDArray[RES_DTYPE]


@attrs.frozen
class ElementNA:
    topology: Topology
    values: npt.NDArray[RES_DTYPE]


@attrs.frozen
class Quantity:
    key: ResultsKey
    ip: tuple[ElementIP, ...]
    nodal: tuple[ElementNA, ...]


@attrs.frozen
class LoadCase:
    name: str
    quantities: tuple[Quantity, ...]
