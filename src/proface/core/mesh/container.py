# SPDX-FileCopyrightText: 2026 Stefano Miccoli <stefano.miccoli@polimi.it>
# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import enum
import functools
from collections.abc import Iterable, Mapping
from typing import Any, overload

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


@overload
def _to_tuple(val: Iterable[Elements]) -> tuple[Elements, ...]: ...


@overload
def _to_tuple(val: Iterable[Set]) -> tuple[Set, ...]: ...


def _to_tuple[T](val: Iterable[T]) -> tuple[T, ...]:
    return tuple(val)


_cmp_numpy = attrs.cmp_using(eq=np.array_equal)


def _attribute_name(instance: object, attribute: attrs.Attribute[Any]) -> str:
    return f"{instance.__class__.__name__}.{attribute.name}"


class _CheckNDIM:
    def __init__(self, ndim: int) -> None:
        self.ndim = ndim

    def __call__(
        self,
        instance: object,
        attribute: attrs.Attribute[npt.NDArray[MESH_IDS]],
        value: npt.NDArray[MESH_IDS],
    ) -> None:
        name = _attribute_name(instance, attribute)
        if value.ndim != self.ndim:
            msg = f"{name} must be {self.ndim}-dimensional"
            raise ValueError(msg)


class _CheckIterOf:
    def __init__(self, typ: type) -> None:
        self.typ = typ

    def __call__(
        self, _instance: object, _attribute: object, value: Iterable[Any]
    ) -> None:
        if any(not isinstance(x, self.typ) for x in value):
            msg = f"Not a {self.typ!s}"
            raise TypeError(msg)


def _check_sorted(
    instance: object,
    attribute: attrs.Attribute[npt.NDArray[MESH_IDS]],
    value: npt.NDArray[MESH_IDS],
) -> None:
    name = _attribute_name(instance, attribute)
    if np.any(value[:-1] >= value[1:]):
        msg = f"{name} must be strongly sorted (no duplicates)"
        raise ValueError(msg)


def _check_unique(
    instance: object,
    attribute: attrs.Attribute[npt.NDArray[MESH_IDS]],
    value: npt.NDArray[MESH_IDS],
) -> None:
    name = _attribute_name(instance, attribute)
    if len(np.unique_values(value)) != len(value):
        msg = f"{name} must be unique"
        raise ValueError(msg)


def _check_coordinates(
    instance: Nodes,
    _attribute: attrs.Attribute[COORDINATES],
    value: npt.NDArray[MESH_COORDINATES],
) -> None:
    if value.shape != (len(instance.numbers), DIM):
        msg = "Node coordinates must have shape (number of nodes, DIM)"
        raise ValueError(msg)


def _check_incidences(
    instance: Elements,
    _attribute: attrs.Attribute[IDS_2D],
    value: npt.NDArray[MESH_IDS],
) -> None:
    if len(value) != len(instance.numbers):
        msg = "Element numbers and incidences must have the same length"
        raise ValueError(msg)
    if value.shape[1] not in Topology:
        msg = "Unknown element topology"
        raise ValueError(msg)


def _check_mesh_elements(
    instance: Mesh,
    _attribute: attrs.Attribute[tuple[Elements, ...]],
    value: tuple[Elements, ...],
) -> None:

    # empty tuple is valid
    if not value:
        return

    # validate tuple elements
    topologies = []
    for g in value:
        if not isinstance(g, Elements):
            msg = f"Not an Elements isinstance: {g}"
            raise TypeError(msg)
        if g.topology in topologies:
            msg = f"Repeated topology: {g.topology.name}"
            raise ValueError(msg)
        topologies.append(g.topology)
        if not np.isin(g.incidences, instance.nodes.numbers).all():
            msg = f"Elements ({g.topology.name}) reference unknown nodes"
            raise ValueError(msg)
    numbers = np.concat([g.numbers for g in value])
    if len(np.unique_values(numbers)) != len(numbers):
        msg = "Element numbers are not unique"
        raise ValueError(msg)


@attrs.frozen(kw_only=True)
class Nodes:
    """container for mesh nodes: numbers (aka labels) and coordinates"""

    numbers: IDS_1D = attrs.field(
        converter=_to_numbers,
        eq=_cmp_numpy,
        validator=[_CheckNDIM(1), _check_sorted],
    )
    coordinates: COORDINATES = attrs.field(
        converter=_to_coordinates,
        eq=_cmp_numpy,
        validator=_check_coordinates,
    )

    @classmethod
    def from_container(cls, container: Group) -> Nodes:
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

    def __len__(self) -> int:
        return len(self.numbers)

    def __str__(self) -> str:
        return f"Nodes ({len(self):_d})"


@attrs.frozen(kw_only=True)
class Elements:
    """container for same topology elements"""

    numbers: IDS_1D = attrs.field(
        converter=_to_numbers,
        eq=_cmp_numpy,
        validator=[_CheckNDIM(1), _check_unique],
    )
    incidences: IDS_2D = attrs.field(
        converter=_to_numbers,
        eq=_cmp_numpy,
        validator=[_CheckNDIM(2), _check_incidences],
    )

    @property
    def topology(self) -> Topology:
        return Topology(self.incidences.shape[1])

    @functools.cached_property
    def nodes(self) -> IDS_1D:
        return np.unique(self.incidences)

    @classmethod
    def from_container(cls, container: Group) -> Elements:
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

    def __len__(self) -> int:
        return len(self.numbers)

    def __str__(self) -> str:
        return f"Elements ({len(self):_d} {self.topology.name})"


@attrs.frozen(kw_only=True)
class Mesh:
    """container for mesh data"""

    nodes: Nodes
    elements: tuple[Elements, ...] = attrs.field(
        converter=_to_tuple,
        validator=_check_mesh_elements,
    )

    @property
    def elements_dict(self) -> dict[Topology, Elements]:
        return {e.topology: e for e in self.elements}

    @classmethod
    def from_container(cls, container: Group) -> Mesh:
        if not isinstance(container, Mapping):
            msg = f"Argument '{container!r}' is not a Mapping"
            raise TypeError(msg)

        nodes = Nodes.from_container(_group(container, "nodes"))
        elements = tuple(
            Elements.from_container(g)
            for g in _group(container, "elements").values()
        )

        return cls(
            nodes=nodes,
            elements=elements,
        )

    def __str__(self) -> str:
        return f"Mesh: {self.nodes}; {', '.join(str(g) for g in self.elements)}"


@attrs.frozen(kw_only=True)
class Set:
    """container for named sets"""

    name: str
    members: IDS_1D = attrs.field(
        converter=_to_numbers,
        eq=_cmp_numpy,
        validator=[_CheckNDIM(1), _check_sorted],
    )

    def __len__(self) -> int:
        return len(self.members)

    def __str__(self) -> str:
        return f"Set ({self.name}: {len(self):_d})"


@attrs.frozen(kw_only=True)
class NamedRegion:
    """named sets"""

    sets_element: tuple[Set, ...] = attrs.field(
        default=(),
        converter=_to_tuple,
        validator=_CheckIterOf(Set),
    )
    sets_node: tuple[Set, ...] = attrs.field(
        default=(),
        converter=_to_tuple,
        validator=_CheckIterOf(Set),
    )

    @classmethod
    def from_container(cls, container: Group) -> NamedRegion:
        if not isinstance(container, Mapping):
            msg = f"Argument '{container!r}' is not a Mapping"
            raise TypeError(msg)

        g = _group(container, "sets/element")
        sets_element = tuple(
            Set(name=k, members=_array(g, k, dtype=MESH_IDS)) for k in g
        )
        g = _group(container, "sets/node")
        sets_node = tuple(
            Set(name=k, members=_array(g, k, dtype=MESH_IDS)) for k in g
        )

        return cls(
            sets_element=sets_element,
            sets_node=sets_node,
        )
