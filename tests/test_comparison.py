# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

import numpy as np

from proface.core.mesh import DIM, Elements, Mesh, Nodes


def test_nodes_compare_by_array_values() -> None:
    nodes = Nodes(
        numbers=np.array([1, 2], dtype=np.uint64),
        coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
    )
    same = Nodes(numbers=[1, 2], coordinates=[[0, 0, 0], [1, 0, 0]])
    different_numbers = Nodes(
        numbers=[1, 3],
        coordinates=[[0, 0, 0], [1, 0, 0]],
    )
    different_coordinates = Nodes(
        numbers=[1, 2],
        coordinates=[[0, 0, 0], [2, 0, 0]],
    )

    assert nodes == same
    assert same == nodes
    assert nodes != different_numbers
    assert nodes != different_coordinates
    assert nodes != object()


def test_elements_compare_by_topology_and_array_values() -> None:
    elements = Elements(
        numbers=np.array([10, 11], dtype=np.uint64),
        incidences=np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5]],
            dtype=np.uint64,
        ),
    )
    same = Elements(
        numbers=[10, 11],
        incidences=[[1, 2, 3, 4], [2, 3, 4, 5]],
    )
    different_numbers = Elements(
        numbers=[10, 12],
        incidences=[[1, 2, 3, 4], [2, 3, 4, 5]],
    )
    different_incidences = Elements(
        numbers=[10, 11],
        incidences=[[1, 2, 3, 4], [2, 3, 4, 6]],
    )
    different_topology = Elements(
        numbers=[10],
        incidences=[[1, 2, 3, 4, 5]],
    )

    assert elements == same
    assert same == elements
    assert elements != different_numbers
    assert elements != different_incidences
    assert elements != different_topology
    assert elements != object()


def test_mesh_compare_by_nested_container_values() -> None:
    mesh = Mesh(
        nodes=Nodes(numbers=[1, 2, 3, 4, 5], coordinates=np.zeros((5, DIM))),
        elements=(
            Elements(
                numbers=[10],
                incidences=[[1, 2, 3, 4]],
            ),
        ),
    )
    same = Mesh(
        nodes=Nodes(
            numbers=np.array([1, 2, 3, 4, 5], dtype=np.uint64),
            coordinates=np.zeros((5, DIM), dtype=np.float64),
        ),
        elements=(
            Elements(
                numbers=np.array([10], dtype=np.uint64),
                incidences=np.array([[1, 2, 3, 4]], dtype=np.uint64),
            ),
        ),
    )
    different_nodes = Mesh(
        nodes=Nodes(numbers=[1, 2, 3, 4, 5], coordinates=np.ones((5, DIM))),
        elements=mesh.elements,
    )
    different_elements = Mesh(
        nodes=mesh.nodes,
        elements=(
            Elements(
                numbers=[11],
                incidences=[[1, 2, 3, 4]],
            ),
        ),
    )

    assert mesh == same
    assert same == mesh
    assert mesh != different_nodes
    assert mesh != different_elements
    assert mesh != object()
