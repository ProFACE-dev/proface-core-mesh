# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from proface.core.mesh import DIM, Elements, Mesh, Nodes, Topology


def test_nodes_direct_initialization_converts_valid_inputs() -> None:
    nodes = Nodes(ids=[1, 2], coord=[[0, 0, 0], [1, 0, 0]])

    assert nodes.ids.dtype == np.uint32
    assert nodes.coord.dtype == np.float32
    assert nodes.coord.shape == (2, DIM)
    np.testing.assert_array_equal(nodes.ids, [1, 2])
    np.testing.assert_array_equal(
        nodes.coord,
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
    )


def test_nodes_direct_initialization_accepts_empty_nodes() -> None:
    nodes = Nodes(ids=(), coord=np.array(()).reshape((0, DIM)))

    assert nodes.ids.dtype == np.uint32
    assert nodes.coord.dtype == np.float32
    assert nodes.coord.shape == (0, DIM)
    np.testing.assert_array_equal(nodes.ids, [])


def test_nodes_direct_initialization_rejects_invalid_ids_shape() -> None:
    with pytest.raises(ValueError, match=r"Nodes\.ids must be 1-dimensional"):
        Nodes(ids=[[1, 2]], coord=[[0, 0, 0]])


def test_nodes_direct_initialization_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match=r"Nodes\.ids must be unique"):
        Nodes(ids=[1, 1], coord=[[0, 0, 0], [1, 0, 0]])


def test_nodes_direct_initialization_rejects_invalid_coord_shape() -> None:
    match = r"Node coordinates must have shape \(number of nodes, DIM\)"
    with pytest.raises(ValueError, match=match):
        Nodes(ids=[1, 2], coord=[[0, 0, 0]])


def test_elements_direct_initialization_converts_valid_inputs() -> None:
    elements = Elements(
        topology=Topology.C3D4,
        ids=[10, 11],
        incidences=[[1, 2, 3, 4], [2, 3, 4, 5]],
    )

    assert elements.topology is Topology.C3D4
    assert elements.ids.dtype == np.uint32
    assert elements.incidences.dtype == np.uint32
    np.testing.assert_array_equal(elements.ids, [10, 11])
    np.testing.assert_array_equal(
        elements.incidences,
        np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.uint32),
    )
    np.testing.assert_array_equal(elements.nodes, [1, 2, 3, 4, 5])


def test_elements_direct_initialization_accepts_empty_elements() -> None:
    elements = Elements(
        topology=Topology.C3D4,
        ids=(),
        incidences=np.array(()).reshape((0, Topology.C3D4.value)),
    )

    assert elements.ids.dtype == np.uint32
    assert elements.incidences.dtype == np.uint32
    assert elements.incidences.shape == (0, Topology.C3D4.value)
    np.testing.assert_array_equal(elements.nodes, [])


def test_elements_direct_initialization_rejects_invalid_ids_shape() -> None:
    with pytest.raises(
        ValueError, match=r"Elements\.ids must be 1-dimensional"
    ):
        Elements(
            topology=Topology.C3D4,
            ids=[[10, 11]],
            incidences=[[1, 2, 3, 4]],
        )


def test_elements_direct_initialization_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match=r"Elements\.ids must be unique"):
        Elements(
            topology=Topology.C3D4,
            ids=[10, 10],
            incidences=[[1, 2, 3, 4], [2, 3, 4, 5]],
        )


def test_elements_direct_initialization_rejects_incidences_shape() -> None:
    match = "Element incidences must be 2-dimensional"
    with pytest.raises(ValueError, match=match):
        Elements(
            topology=Topology.C3D4,
            ids=[10],
            incidences=[1, 2, 3, 4],
        )


def test_elements_direct_initialization_rejects_length_mismatch() -> None:
    match = "Element ids and incidences must have the same length"
    with pytest.raises(ValueError, match=match):
        Elements(
            topology=Topology.C3D4,
            ids=[10, 11],
            incidences=[[1, 2, 3, 4]],
        )


def test_elements_direct_initialization_rejects_topology_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="Element incidences do not match topology",
    ):
        Elements(
            topology=Topology.C3D5,
            ids=[10],
            incidences=[[1, 2, 3, 4]],
        )


def test_mesh_direct_initialization_accepts_valid_elements() -> None:
    nodes = Nodes(ids=[1, 2, 3, 4, 5], coord=np.zeros((5, DIM)))
    c3d4 = Elements(
        topology=Topology.C3D4,
        ids=[10],
        incidences=[[1, 2, 3, 4]],
    )
    c3d5 = Elements(
        topology=Topology.C3D5,
        ids=[20],
        incidences=[[1, 2, 3, 4, 5]],
    )

    mesh = Mesh(nodes=nodes, elements=(c3d4, c3d5))

    assert mesh.nodes is nodes
    assert mesh.elements == (c3d4, c3d5)


def test_mesh_direct_initialization_accepts_empty_elements() -> None:
    nodes = Nodes(ids=[1], coord=[[0, 0, 0]])

    mesh = Mesh(nodes=nodes, elements=())

    assert mesh.nodes is nodes
    assert mesh.elements == ()


def test_mesh_direct_initialization_rejects_unknown_node_references() -> None:
    nodes = Nodes(ids=[1, 2, 3], coord=np.zeros((3, DIM)))
    elements = Elements(
        topology=Topology.C3D4,
        ids=[10],
        incidences=[[1, 2, 3, 4]],
    )

    match = r"Elements \(C3D4\) reference unknown nodes"
    with pytest.raises(ValueError, match=match):
        Mesh(nodes=nodes, elements=(elements,))


def test_mesh_direct_initialization_rejects_duplicate_element_ids() -> None:
    nodes = Nodes(ids=[1, 2, 3, 4, 5], coord=np.zeros((5, DIM)))
    c3d4 = Elements(
        topology=Topology.C3D4,
        ids=[10],
        incidences=[[1, 2, 3, 4]],
    )
    c3d5 = Elements(
        topology=Topology.C3D5,
        ids=[10],
        incidences=[[1, 2, 3, 4, 5]],
    )

    with pytest.raises(ValueError, match="Element numbers are not unique"):
        Mesh(nodes=nodes, elements=(c3d4, c3d5))


def test_mesh_direct_initialization_accepts_empty_nodes_and_elements() -> None:
    nodes = Nodes(ids=(), coord=np.array(()).reshape((0, DIM)))

    mesh = Mesh(nodes=nodes, elements=())

    assert mesh.nodes is nodes
    assert mesh.elements == ()
