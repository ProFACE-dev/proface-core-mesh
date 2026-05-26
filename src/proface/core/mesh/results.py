# SPDX-FileCopyrightText: 2026 Stefano Miccoli <stefano.miccoli@polimi.it>
# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, auto

import attrs
import attrs.validators
import numpy as np
import numpy.typing as npt

from ._constants import DIM
from .container import Topology
from .helpers import Group, array, group

RES_DTYPE = np.float32
RES_ARRAY = npt.NDArray[RES_DTYPE]

SCALAR = ()
VECTOR = (DIM,)
TENSOR = (DIM * (DIM + 1) // 2,)


class QuantityKey(Enum):
    S = auto()
    SP = auto()
    COORD = auto()
    IVOL = auto()


RES_DIM = {
    QuantityKey.S: TENSOR,
    QuantityKey.SP: VECTOR,
    QuantityKey.COORD: VECTOR,
    QuantityKey.IVOL: SCALAR,
}


@attrs.frozen
class Quantity:
    key: QuantityKey = attrs.field(
        validator=attrs.validators.instance_of(QuantityKey)
    )
    ip: dict[Topology, RES_ARRAY]
    nodal: dict[Topology, RES_ARRAY]

    @classmethod
    def from_group(cls, key: QuantityKey, group: Group) -> Quantity:
        if not isinstance(group, Mapping):
            msg = f"Argument '{group!r}' is not a Group"
            raise TypeError(msg)

        return cls(
            key=key,
            ip=_get_location("integration_point", group),
            nodal=_get_location("nodal_averaged", group),
        )

    def __str__(self) -> str:
        locs = []
        if self.ip:
            locs.append("ip")
        if self.nodal:
            locs.append("nodal")
        return f"{self.key!s}({', '.join(locs)})"


@attrs.frozen
class LoadCase:
    name: str
    quantities: tuple[Quantity, ...]

    @classmethod
    def from_group(cls, name: str, group: Group) -> LoadCase:
        if not isinstance(group, Mapping):
            msg = f"Argument '{group!r}' is not a Group"
            raise TypeError(msg)

        return cls(
            name=name,
            quantities=tuple(
                Quantity.from_group(key=QuantityKey[key], group=group)
                for key, group in group.items()
            ),
        )

    def __str__(self) -> str:
        return f"{self.name}: {', '.join(str(q) for q in self.quantities)}"


def _get_location(key: str, container: Group) -> dict[Topology, RES_ARRAY]:

    try:
        loc = group(container=container, key=key)
    except ValueError:
        return {}
    else:
        return {
            Topology[name]: array(container=loc, key=name, dtype=RES_DTYPE)
            for name in loc
        }
