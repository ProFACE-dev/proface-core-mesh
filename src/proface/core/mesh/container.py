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
MESH_COORDINATES = np.float32

type Group = Mapping[str, Any]
type IDS_1D = np.ndarray[tuple[int], np.dtype[MESH_IDS]]
type IDS_2D = np.ndarray[tuple[int, int], np.dtype[MESH_IDS]]
type COORDINATES = np.ndarray[tuple[int, int], np.dtype[MESH_COORDINATES]]


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


def _to_numbers(val: npt.ArrayLike) -> npt.NDArray[MESH_IDS]:
    return np.asarray(val, dtype=MESH_IDS)


def _to_coordinates(val: npt.ArrayLike) -> npt.NDArray[MESH_COORDINATES]:
    return np.asarray(val, dtype=MESH_COORDINATES)


_cmp_numpy = attrs.cmp_using(eq=np.array_equal)


@attrs.frozen(kw_only=True)
class _Numbered:
    """container for numbered entities"""

    numbers: IDS_1D = attrs.field(converter=_to_numbers, eq=_cmp_numpy)

    @numbers.validator
    def _check_numbers(
        self, _attribute: object, value: npt.NDArray[MESH_IDS]
    ) -> None:
        name = self.__class__.__name__
        if value.ndim != 1:
            msg = f"{name}.numbers must be 1-dimensional"
            raise ValueError(msg)
        if len(np.unique_values(value)) != len(value):
            msg = f"{name}.numbers must be unique"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.numbers)


@attrs.frozen(kw_only=True)
class Nodes(_Numbered):
    """container for mesh nodes: numbers (aka labels) and coordinates"""

    coordinates: COORDINATES = attrs.field(
        converter=_to_coordinates,
        eq=_cmp_numpy,
    )

    @coordinates.validator
    def _check_coordinates(
        self, _attribute: object, value: npt.NDArray[MESH_COORDINATES]
    ) -> None:
        if value.shape != (len(self.numbers), DIM):
            msg = "Node coordinates must have shape (number of nodes, DIM)"
            raise ValueError(msg)

    @classmethod
    def from_container(cls, container: Group) -> "Nodes":
        if not isinstance(container, Mapping):
            msg = f"Argument '{container!r}' is not a Mapping"
            raise TypeError(msg)
        return cls(
            numbers=_array(container, "numbers", dtype=MESH_IDS),
            coordinates=_array(
                container,
                "coordinates",
                dtype=MESH_COORDINATES,
            ),
        )

    def __str__(self) -> str:
        return f"Nodes ({len(self):_d})"


@attrs.frozen(kw_only=True)
class Elements(_Numbered):
    """container for same topology elements"""

    incidences: IDS_2D = attrs.field(converter=_to_numbers, eq=_cmp_numpy)

    @incidences.validator
    def _check_incidences(
        self, _attribute: object, value: npt.NDArray[MESH_IDS]
    ) -> None:
        if value.ndim != 2:  # noqa: PLR2004
            msg = "Element incidences must be 2-dimensional"
            raise ValueError(msg)
        if len(value) != len(self.numbers):
            msg = "Element numbers and incidences must have the same length"
            raise ValueError(msg)
        if value.shape[1] not in Topology:
            msg = "Unknown element topology"
            raise ValueError(msg)

    @property
    def topology(self) -> Topology:
        return Topology(self.incidences.shape[1])

    @functools.cached_property
    def nodes(self) -> IDS_1D:
        return np.unique(self.incidences)

    @classmethod
    def from_container(cls, container: Group) -> "Elements":
        if not isinstance(container, Mapping):
            msg = f"Argument '{container!r}' is not a Mapping"
            raise TypeError(msg)
        numbers = _array(container, "numbers", dtype=MESH_IDS)
        incidences = _array(container, "incidences", dtype=MESH_IDS)
        els = cls(numbers=numbers, incidences=incidences)

        nodes = _array(container, "nodes", dtype=MESH_IDS)
        if not np.array_equal(nodes, els.nodes):
            msg = "Element nodes are inconsistent with incidences"
            raise ValueError(msg)

        return els

    def __str__(self) -> str:
        return f"Elements ({len(self):_d} {self.topology.name})"


@attrs.frozen(kw_only=True)
class Mesh:
    """container for mesh data"""

    nodes: Nodes
    elements: tuple[Elements, ...] = attrs.field()

    @elements.validator
    def _check_elements(
        self, _attribute: object, value: tuple[Elements, ...]
    ) -> None:
        if not value:
            return

        topologies = []
        for g in value:
            if g.topology in topologies:
                msg = f"Repeated topology: {g.topology.name}"
                raise ValueError(msg)
            topologies.append(g.topology)
            if not np.isin(g.incidences, self.nodes.numbers).all():
                msg = f"Elements ({g.topology.name}) reference unknown nodes"
                raise ValueError(msg)
        numbers = np.concat([g.numbers for g in value])
        if len(np.unique_values(numbers)) != len(numbers):
            msg = "Element numbers are not unique"
            raise ValueError(msg)

    @property
    def elements_dict(self) -> dict[Topology, Elements]:
        return {e.topology: e for e in self.elements}

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

    def __str__(self) -> str:
        return f"Mesh: {self.nodes}; {', '.join(str(g) for g in self.elements)}"
