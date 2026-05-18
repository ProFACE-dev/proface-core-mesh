# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

import numpy as np

from proface.core.mesh import DIM, Elements, Mesh, Nodes, Topology


def test_nodes_compare_by_array_values() -> None:
    nodes = Nodes(
        ids=np.array([1, 2], dtype=np.uint64),
        coord=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
    )
    same = Nodes(ids=[1, 2], coord=[[0, 0, 0], [1, 0, 0]])
    different_ids = Nodes(ids=[1, 3], coord=[[0, 0, 0], [1, 0, 0]])
    different_coord = Nodes(ids=[1, 2], coord=[[0, 0, 0], [2, 0, 0]])

    assert nodes == same
    assert same == nodes
    assert nodes != different_ids
    assert nodes != different_coord
    assert nodes != object()


def test_elements_compare_by_topology_and_array_values() -> None:
    elements = Elements(
        topology=Topology.C3D4,
        ids=np.array([10, 11], dtype=np.uint64),
        incidences=np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5]],
            dtype=np.uint64,
        ),
    )
    same = Elements(
        topology=Topology.C3D4,
        ids=[10, 11],
        incidences=[[1, 2, 3, 4], [2, 3, 4, 5]],
    )
    different_ids = Elements(
        topology=Topology.C3D4,
        ids=[10, 12],
        incidences=[[1, 2, 3, 4], [2, 3, 4, 5]],
    )
    different_incidences = Elements(
        topology=Topology.C3D4,
        ids=[10, 11],
        incidences=[[1, 2, 3, 4], [2, 3, 4, 6]],
    )
    different_topology = Elements(
        topology=Topology.C3D5,
        ids=[10],
        incidences=[[1, 2, 3, 4, 5]],
    )

    assert elements == same
    assert same == elements
    assert elements != different_ids
    assert elements != different_incidences
    assert elements != different_topology
    assert elements != object()


def test_mesh_compare_by_nested_container_values() -> None:
    mesh = Mesh(
        nodes=Nodes(ids=[1, 2, 3, 4, 5], coord=np.zeros((5, DIM))),
        elements=(
            Elements(
                topology=Topology.C3D4,
                ids=[10],
                incidences=[[1, 2, 3, 4]],
            ),
        ),
    )
    same = Mesh(
        nodes=Nodes(
            ids=np.array([1, 2, 3, 4, 5], dtype=np.uint64),
            coord=np.zeros((5, DIM), dtype=np.float64),
        ),
        elements=(
            Elements(
                topology=Topology.C3D4,
                ids=np.array([10], dtype=np.uint64),
                incidences=np.array([[1, 2, 3, 4]], dtype=np.uint64),
            ),
        ),
    )
    different_nodes = Mesh(
        nodes=Nodes(ids=[1, 2, 3, 4, 5], coord=np.ones((5, DIM))),
        elements=mesh.elements,
    )
    different_elements = Mesh(
        nodes=mesh.nodes,
        elements=(
            Elements(
                topology=Topology.C3D4,
                ids=[11],
                incidences=[[1, 2, 3, 4]],
            ),
        ),
    )

    assert mesh == same
    assert same == mesh
    assert mesh != different_nodes
    assert mesh != different_elements
    assert mesh != object()
