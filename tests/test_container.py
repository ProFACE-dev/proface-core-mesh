# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

from typing import Any, cast

import numpy as np
import pytest

from proface.core.mesh import DIM, Elements, Mesh, Nodes, Topology


def test_nodes_direct_initialization_converts_valid_inputs() -> None:
    nodes = Nodes(numbers=[1, 2], coordinates=[[0, 0, 0], [1, 0, 0]])

    assert nodes.numbers.dtype == np.uint32
    assert nodes.coordinates.dtype == np.float32
    assert nodes.coordinates.shape == (2, DIM)
    np.testing.assert_array_equal(nodes.numbers, [1, 2])
    np.testing.assert_array_equal(
        nodes.coordinates,
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
    )


def test_nodes_direct_initialization_accepts_empty_nodes() -> None:
    nodes = Nodes(numbers=(), coordinates=np.array(()).reshape((0, DIM)))

    assert nodes.numbers.dtype == np.uint32
    assert nodes.coordinates.dtype == np.float32
    assert nodes.coordinates.shape == (0, DIM)
    np.testing.assert_array_equal(nodes.numbers, [])


def test_nodes_direct_initialization_supports_len() -> None:
    numbers = [1, 2]
    nodes = Nodes(numbers=numbers, coordinates=[[0, 0, 0], [1, 0, 0]])

    assert len(nodes) == len(numbers)


def test_nodes_direct_initialization_supports_str() -> None:
    nodes = Nodes(numbers=[1], coordinates=[[0, 0, 0]])

    assert isinstance(str(nodes), str)


def test_nodes_direct_initialization_rejects_positional_arguments() -> None:
    with pytest.raises(TypeError, match="positional"):
        cast("Any", Nodes)([1], [[0, 0, 0]])


def test_nodes_direct_initialization_rejects_invalid_numbers_shape() -> None:
    with pytest.raises(
        ValueError,
        match=r"Nodes\.numbers must be 1-dimensional",
    ):
        Nodes(numbers=[[1, 2]], coordinates=[[0, 0, 0]])


def test_nodes_direct_initialization_rejects_duplicate_numbers() -> None:
    with pytest.raises(
        ValueError,
        match=r"Nodes\.numbers must be strongly sorted",
    ):
        Nodes(numbers=[1, 1], coordinates=[[0, 0, 0], [1, 0, 0]])


def test_nodes_direct_initialization_rejects_unsorted_numbers() -> None:
    with pytest.raises(
        ValueError,
        match=r"Nodes\.numbers must be strongly sorted",
    ):
        Nodes(numbers=[2, 1], coordinates=[[1, 0, 0], [0, 0, 0]])


def test_nodes_direct_initialization_rejects_bad_coordinates_shape() -> None:
    match = r"Node coordinates must have shape \(number of nodes, DIM\)"
    with pytest.raises(ValueError, match=match):
        Nodes(numbers=[1, 2], coordinates=[[0, 0, 0]])


def test_elements_direct_initialization_converts_valid_inputs() -> None:
    elements = Elements(
        numbers=[10, 11],
        incidences=[[1, 2, 3, 4], [2, 3, 4, 5]],
    )

    assert elements.topology is Topology.C3D4
    assert elements.numbers.dtype == np.uint32
    assert elements.incidences.dtype == np.uint32
    np.testing.assert_array_equal(elements.numbers, [10, 11])
    np.testing.assert_array_equal(
        elements.incidences,
        np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.uint32),
    )
    np.testing.assert_array_equal(elements.nodes, [1, 2, 3, 4, 5])


def test_elements_direct_initialization_accepts_empty_elements() -> None:
    elements = Elements(
        numbers=(),
        incidences=np.array(()).reshape((0, Topology.C3D4.value)),
    )

    assert elements.numbers.dtype == np.uint32
    assert elements.incidences.dtype == np.uint32
    assert elements.incidences.shape == (0, Topology.C3D4.value)
    np.testing.assert_array_equal(elements.nodes, [])


def test_elements_direct_initialization_supports_len() -> None:
    numbers = [10, 11]
    elements = Elements(
        numbers=numbers,
        incidences=[[1, 2, 3, 4], [2, 3, 4, 5]],
    )

    assert len(elements) == len(numbers)


def test_elements_direct_initialization_supports_str() -> None:
    elements = Elements(
        numbers=[10],
        incidences=[[1, 2, 3, 4]],
    )

    assert isinstance(str(elements), str)


def test_elements_direct_initialization_rejects_positional_arguments() -> None:
    with pytest.raises(TypeError, match="positional"):
        cast("Any", Elements)([10], [[1, 2, 3, 4]])


def test_elements_direct_initialization_rejects_invalid_numbers_shape() -> None:
    with pytest.raises(
        ValueError, match=r"Elements\.numbers must be 1-dimensional"
    ):
        Elements(
            numbers=[[10, 11]],
            incidences=[[1, 2, 3, 4]],
        )


def test_elements_direct_initialization_rejects_duplicate_numbers() -> None:
    with pytest.raises(
        ValueError,
        match=r"Elements\.numbers must be unique",
    ):
        Elements(
            numbers=[10, 10],
            incidences=[[1, 2, 3, 4], [2, 3, 4, 5]],
        )


def test_elements_direct_initialization_accepts_unsorted_numbers() -> None:
    elements = Elements(
        numbers=[11, 10],
        incidences=[[2, 3, 4, 5], [1, 2, 3, 4]],
    )

    np.testing.assert_array_equal(elements.numbers, [11, 10])


def test_elements_direct_initialization_rejects_incidences_shape() -> None:
    match = r"Elements\.incidences must be 2-dimensional"
    with pytest.raises(ValueError, match=match):
        Elements(
            numbers=[10],
            incidences=[1, 2, 3, 4],
        )


def test_elements_direct_initialization_rejects_length_mismatch() -> None:
    match = "Element numbers and incidences must have the same length"
    with pytest.raises(ValueError, match=match):
        Elements(
            numbers=[10, 11],
            incidences=[[1, 2, 3, 4]],
        )


def test_elements_direct_initialization_rejects_unsupported_topology() -> None:
    with pytest.raises(ValueError, match="Unknown element topology"):
        Elements(
            numbers=[10],
            incidences=[[1, 2, 3]],
        )


def test_mesh_direct_initialization_accepts_valid_elements() -> None:
    nodes = Nodes(numbers=[1, 2, 3, 4, 5], coordinates=np.zeros((5, DIM)))
    c3d4 = Elements(
        numbers=[10],
        incidences=[[1, 2, 3, 4]],
    )
    c3d5 = Elements(
        numbers=[20],
        incidences=[[1, 2, 3, 4, 5]],
    )

    mesh = Mesh(nodes=nodes, elements=(c3d4, c3d5))

    assert mesh.nodes is nodes
    assert mesh.elements == (c3d4, c3d5)


def test_mesh_direct_initialization_accepts_empty_elements() -> None:
    nodes = Nodes(numbers=[1], coordinates=[[0, 0, 0]])

    mesh = Mesh(nodes=nodes, elements=())

    assert mesh.nodes is nodes
    assert mesh.elements == ()


def test_mesh_direct_initialization_converts_elements_to_tuple() -> None:
    nodes = Nodes(numbers=[1, 2, 3, 4], coordinates=np.zeros((4, DIM)))
    elements = Elements(
        numbers=[10],
        incidences=[[1, 2, 3, 4]],
    )

    mesh = Mesh(nodes=nodes, elements=[elements])

    assert mesh.elements == (elements,)


def test_mesh_elements_dict_indexes_elements_by_topology() -> None:
    nodes = Nodes(numbers=[1, 2, 3, 4, 5], coordinates=np.zeros((5, DIM)))
    c3d4 = Elements(
        numbers=[10],
        incidences=[[1, 2, 3, 4]],
    )
    c3d5 = Elements(
        numbers=[20],
        incidences=[[1, 2, 3, 4, 5]],
    )

    mesh = Mesh(nodes=nodes, elements=(c3d4, c3d5))

    assert mesh.elements_dict == {
        Topology.C3D4: c3d4,
        Topology.C3D5: c3d5,
    }


def test_mesh_direct_initialization_supports_str() -> None:
    nodes = Nodes(numbers=[1, 2, 3, 4], coordinates=np.zeros((4, DIM)))
    elements = Elements(
        numbers=[10],
        incidences=[[1, 2, 3, 4]],
    )
    mesh = Mesh(nodes=nodes, elements=(elements,))

    assert isinstance(str(mesh), str)


def test_mesh_direct_initialization_rejects_positional_arguments() -> None:
    nodes = Nodes(numbers=[1], coordinates=[[0, 0, 0]])

    with pytest.raises(TypeError, match="positional"):
        cast("Any", Mesh)(nodes, ())


def test_mesh_direct_initialization_rejects_non_elements() -> None:
    nodes = Nodes(numbers=[1], coordinates=[[0, 0, 0]])

    with pytest.raises(TypeError, match="Not an Elements isinstance"):
        Mesh(nodes=nodes, elements=cast("Any", (object(),)))


def test_mesh_direct_initialization_rejects_unknown_node_references() -> None:
    nodes = Nodes(numbers=[1, 2, 3], coordinates=np.zeros((3, DIM)))
    elements = Elements(
        numbers=[10],
        incidences=[[1, 2, 3, 4]],
    )

    match = r"Elements \(C3D4\) reference unknown nodes"
    with pytest.raises(ValueError, match=match):
        Mesh(nodes=nodes, elements=(elements,))


def test_mesh_direct_initialization_rejects_repeated_topologies() -> None:
    nodes = Nodes(numbers=[1, 2, 3, 4, 5], coordinates=np.zeros((5, DIM)))
    first = Elements(
        numbers=[10],
        incidences=[[1, 2, 3, 4]],
    )
    second = Elements(
        numbers=[11],
        incidences=[[2, 3, 4, 5]],
    )

    with pytest.raises(ValueError, match="Repeated topology: C3D4"):
        Mesh(nodes=nodes, elements=(first, second))


def test_mesh_direct_initialization_rejects_duplicate_element_numbers() -> None:
    nodes = Nodes(numbers=[1, 2, 3, 4, 5], coordinates=np.zeros((5, DIM)))
    c3d4 = Elements(
        numbers=[10],
        incidences=[[1, 2, 3, 4]],
    )
    c3d5 = Elements(
        numbers=[10],
        incidences=[[1, 2, 3, 4, 5]],
    )

    with pytest.raises(ValueError, match="Element numbers are not unique"):
        Mesh(nodes=nodes, elements=(c3d4, c3d5))


def test_mesh_direct_initialization_accepts_empty_nodes_and_elements() -> None:
    nodes = Nodes(numbers=(), coordinates=np.array(()).reshape((0, DIM)))

    mesh = Mesh(nodes=nodes, elements=())

    assert mesh.nodes is nodes
    assert mesh.elements == ()
