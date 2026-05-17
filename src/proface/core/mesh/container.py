# SPDX-FileCopyrightText: 2026 Stefano Miccoli <stefano.miccoli@polimi.it>
# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

import enum
import functools
from collections.abc import Mapping
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt

DIM = 3


class Topology(enum.Enum):
    C3D4 = 4
    C3D5 = 5
    C3D6 = 6
    C3D8 = 8
    C3D10 = 10
    C3D13 = 13
    C3D15 = 15
    C3D20 = 20


MESH_IDS = np.uint32
MESH_COORD = np.float32

type Group = Mapping[str, Any]
type IDS_1D = np.ndarray[tuple[int], np.dtype[MESH_IDS]]
type IDS_2D = np.ndarray[tuple[int, int], np.dtype[MESH_IDS]]
type COORD = np.ndarray[tuple[int, int], np.dtype[MESH_COORD]]


def _get(container: Group, key: str) -> Any:
    try:
        return container[key]
    except KeyError as exc:
        msg = f"mesh container missing key: {key!r}"
        raise ValueError(msg) from exc


def _array[T: np.number](
    container: Group, key: str, *, dtype: type[T]
) -> npt.NDArray[T]:
    value = _get(container, key)
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        msg = f"mesh container[{key!r}] must be a numeric array-like dataset"
        raise TypeError(msg) from exc


def _group(container: Group, key: str) -> Group:
    value = _get(container, key)
    if not isinstance(value, Mapping):
        msg = f"mesh container[{key!r}] must be a group"
        raise TypeError(msg)
    return value


def _to_ids(val: npt.ArrayLike) -> npt.NDArray[MESH_IDS]:
    return np.asarray(val, dtype=MESH_IDS)


def _to_coord(val: npt.ArrayLike) -> npt.NDArray[MESH_COORD]:
    return np.asarray(val, dtype=MESH_COORD)


@attrs.frozen
class Nodes:
    """container for mesh nodes: numbers (aka labels) and coordinates"""

    ids: IDS_1D = attrs.field(converter=_to_ids)
    coord: COORD = attrs.field(converter=_to_coord)

    @ids.validator
    def check_ids(
        self, _attribute: object, value: npt.NDArray[MESH_IDS]
    ) -> None:
        if value.ndim != 1:
            msg = "Node ids must be 1-dimensional"
            raise ValueError(msg)
        if len(np.unique_values(value)) != len(value):
            msg = "Node ids must be unique"
            raise ValueError(msg)

    @coord.validator
    def check_coord(
        self, _attribute: object, value: npt.NDArray[MESH_COORD]
    ) -> None:
        if value.shape != (len(self.ids), DIM):
            msg = "Node coordinates must have shape (number of nodes, DIM)"
            raise ValueError(msg)

    @classmethod
    def from_container(cls, container: Group) -> "Nodes":
        if not isinstance(container, Mapping):
            msg = f"Argument '{container!r}' is not a Mapping"
            raise TypeError(msg)
        return cls(
            ids=_array(container, "numbers", dtype=MESH_IDS),
            coord=_array(container, "coordinates", dtype=MESH_COORD),
        )


@attrs.frozen
class Elements:
    """container for same topology elements"""

    topology: Topology
    ids: IDS_1D = attrs.field(converter=_to_ids)
    incidences: IDS_2D = attrs.field(converter=_to_ids)

    @ids.validator
    def check_ids(
        self, _attribute: object, value: npt.NDArray[MESH_IDS]
    ) -> None:
        if value.ndim != 1:
            msg = "Element ids must be 1-dimensional"
            raise ValueError(msg)
        if len(np.unique_values(value)) != len(value):
            msg = "Element ids must be unique"
            raise ValueError(msg)

    @incidences.validator
    def check_incidences(
        self, _attribute: object, value: npt.NDArray[MESH_IDS]
    ) -> None:
        if value.ndim != 2:  # noqa: PLR2004
            msg = "Element incidences must be 2-dimensional"
            raise ValueError(msg)
        if len(value) != len(self.ids):
            msg = "Element ids and incidences must have the same length"
            raise ValueError(msg)
        if value.shape[1] != self.topology.value:
            msg = "Element incidences do not match topology"
            raise ValueError(msg)

    @functools.cached_property
    def nodes(self) -> IDS_1D:
        return np.unique(self.incidences)

    @classmethod
    def from_container(cls, container: Group) -> "Elements":
        if not isinstance(container, Mapping):
            msg = f"Argument '{container!r}' is not a Mapping"
            raise TypeError(msg)
        ids = _array(container, "numbers", dtype=MESH_IDS)
        incidences = _array(container, "incidences", dtype=MESH_IDS)
        if incidences.ndim != 2:  # noqa: PLR2004
            msg = "Element incidences must be 2-dimensional"
            raise ValueError(msg)
        topology = Topology(incidences.shape[1])
        els = cls(topology=topology, ids=ids, incidences=incidences)

        nodes = _array(container, "nodes", dtype=MESH_IDS)
        if not np.array_equal(nodes, els.nodes):
            msg = "Element nodes are inconsistent with incidences"
            raise ValueError(msg)

        return els


@attrs.frozen
class Mesh:
    """container for mesh data"""

    nodes: Nodes
    elements: tuple[Elements, ...] = attrs.field()

    @elements.validator
    def check_elements(
        self, _attribute: object, value: tuple[Elements, ...]
    ) -> None:
        if not value:
            return
        for g in value:
            if not np.isin(g.incidences, self.nodes.ids).all():
                msg = f"Elements ({g.topology.name}) reference unknown nodes"
                raise ValueError(msg)
        ids = np.concat([g.ids for g in value])
        if len(np.unique_values(ids)) != len(ids):
            msg = "Element numbers are not unique"
            raise ValueError(msg)

    @classmethod
    def from_container(cls, container: Group) -> "Mesh":
        if not isinstance(container, Mapping):
            msg = f"Argument '{container!r}' is not a Mapping"
            raise TypeError(msg)
        nodes = Nodes.from_container(_group(container, "nodes"))
        elements = tuple(
            Elements.from_container(g)
            for g in _group(container, "elements").values()
        )

        return cls(nodes=nodes, elements=elements)
