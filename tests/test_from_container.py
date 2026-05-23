# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

from typing import Any, cast

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

from proface.core.mesh import (
    DIM,
    Elements,
    Mesh,
    NamedRegion,
    Nodes,
    Topology,
)


def _nodes_container() -> dict[str, Any]:
    return {
        "numbers": [1, 2, 3, 4, 5],
        "coordinates": np.zeros((5, DIM)),
    }


def _c3d4_container() -> dict[str, Any]:
    return {
        "numbers": [10],
        "incidences": [[1, 2, 3, 4]],
        "nodes": [1, 2, 3, 4],
    }


def _c3d5_container() -> dict[str, Any]:
    return {
        "numbers": [20],
        "incidences": [[1, 2, 3, 4, 5]],
        "nodes": [1, 2, 3, 4, 5],
    }


def test_nodes_from_container_converts_valid_mapping() -> None:
    nodes = Nodes.from_container(_nodes_container())

    assert nodes.numbers.dtype == np.uint32
    assert nodes.coordinates.dtype == np.float32
    assert nodes.coordinates.shape == (5, DIM)
    np.testing.assert_array_equal(nodes.numbers, [1, 2, 3, 4, 5])


def test_nodes_from_container_accepts_empty_nodes() -> None:
    nodes = Nodes.from_container(
        {
            "numbers": [],
            "coordinates": np.array(()).reshape((0, DIM)),
        },
    )

    assert nodes.coordinates.shape == (0, DIM)
    np.testing.assert_array_equal(nodes.numbers, [])


def test_nodes_from_container_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="is not a Mapping"):
        Nodes.from_container(cast("Any", object()))


def test_nodes_from_container_rejects_missing_numbers() -> None:
    with pytest.raises(ValueError, match="missing key: 'numbers'"):
        Nodes.from_container({"coordinates": np.zeros((1, DIM))})


def test_nodes_from_container_rejects_missing_coordinates() -> None:
    with pytest.raises(ValueError, match="missing key: 'coordinates'"):
        Nodes.from_container({"numbers": [1]})


def test_nodes_from_container_rejects_bad_numbers_dataset() -> None:
    match = r"container\['numbers'\].*numeric array-like dataset"
    with pytest.raises(TypeError, match=match):
        Nodes.from_container(
            {"numbers": ["invalid"], "coordinates": np.zeros((1, DIM))},
        )


def test_nodes_from_container_rejects_bad_coordinates_dataset() -> None:
    match = r"container\['coordinates'\].*numeric array-like dataset"
    with pytest.raises(TypeError, match=match):
        Nodes.from_container({"numbers": [1], "coordinates": ["invalid"]})


def test_nodes_from_container_rejects_invalid_schema() -> None:
    match = r"Node coordinates must have shape \(number of nodes, DIM\)"
    with pytest.raises(ValueError, match=match):
        Nodes.from_container({"numbers": [1, 2], "coordinates": [[0, 0, 0]]})


def test_elements_from_container_converts_valid_mapping() -> None:
    elements = Elements.from_container(_c3d4_container())

    assert elements.topology is Topology.C3D4
    assert elements.numbers.dtype == np.uint32
    assert elements.incidences.dtype == np.uint32
    np.testing.assert_array_equal(elements.numbers, [10])
    np.testing.assert_array_equal(elements.incidences, [[1, 2, 3, 4]])
    np.testing.assert_array_equal(elements.nodes, [1, 2, 3, 4])


def test_elements_from_container_accepts_empty_elements() -> None:
    elements = Elements.from_container(
        {
            "numbers": [],
            "incidences": np.array(()).reshape((0, Topology.C3D4.value)),
            "nodes": [],
        },
    )

    assert elements.topology is Topology.C3D4
    assert elements.incidences.shape == (0, Topology.C3D4.value)
    np.testing.assert_array_equal(elements.nodes, [])


def test_elements_from_container_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="is not a Mapping"):
        Elements.from_container(cast("Any", object()))


def test_elements_from_container_rejects_missing_numbers() -> None:
    with pytest.raises(ValueError, match="missing key: 'numbers'"):
        Elements.from_container({"incidences": [[1, 2, 3, 4]], "nodes": []})


def test_elements_from_container_rejects_missing_incidences() -> None:
    with pytest.raises(ValueError, match="missing key: 'incidences'"):
        Elements.from_container({"numbers": [10], "nodes": []})


def test_elements_from_container_rejects_missing_nodes() -> None:
    with pytest.raises(ValueError, match="missing key: 'nodes'"):
        Elements.from_container({"numbers": [10], "incidences": [[1, 2, 3, 4]]})


def test_elements_from_container_rejects_bad_numbers_dataset() -> None:
    match = r"container\['numbers'\].*numeric array-like dataset"
    with pytest.raises(TypeError, match=match):
        Elements.from_container(
            {
                "numbers": ["invalid"],
                "incidences": [[1, 2, 3, 4]],
                "nodes": [1, 2, 3, 4],
            },
        )


def test_elements_from_container_rejects_bad_incidences_dataset() -> None:
    match = r"container\['incidences'\].*numeric array-like dataset"
    with pytest.raises(TypeError, match=match):
        Elements.from_container(
            {
                "numbers": [10],
                "incidences": [["invalid"]],
                "nodes": [1, 2, 3, 4],
            },
        )


def test_elements_from_container_rejects_bad_nodes_dataset() -> None:
    match = r"container\['nodes'\].*numeric array-like dataset"
    with pytest.raises(TypeError, match=match):
        Elements.from_container(
            {
                "numbers": [10],
                "incidences": [[1, 2, 3, 4]],
                "nodes": ["invalid"],
            },
        )


def test_elements_from_container_rejects_incidences_shape() -> None:
    match = r"Elements\.incidences must be 2-dimensional"
    with pytest.raises(ValueError, match=match):
        Elements.from_container(
            {"numbers": [10], "incidences": [1, 2, 3, 4], "nodes": []},
        )


def test_elements_from_container_rejects_unsupported_topology() -> None:
    with pytest.raises(ValueError, match="Unknown element topology"):
        Elements.from_container(
            {"numbers": [10], "incidences": [[1, 2, 3]], "nodes": [1, 2, 3]},
        )


def test_elements_from_container_rejects_length_mismatch() -> None:
    match = "Element numbers and incidences must have the same length"
    with pytest.raises(ValueError, match=match):
        Elements.from_container(
            {
                "numbers": [10, 11],
                "incidences": [[1, 2, 3, 4]],
                "nodes": [1, 2, 3, 4],
            },
        )


def test_elements_from_container_rejects_inconsistent_nodes() -> None:
    match = "Element nodes are inconsistent with incidences"
    with pytest.raises(ValueError, match=match):
        Elements.from_container(
            {
                "numbers": [10],
                "incidences": [[1, 2, 3, 4]],
                "nodes": [1, 2, 3],
            },
        )


def test_mesh_from_container_converts_valid_mapping() -> None:
    mesh = Mesh.from_container(
        {
            "nodes": _nodes_container(),
            "elements": {
                "C3D4": _c3d4_container(),
                "C3D5": _c3d5_container(),
            },
        },
    )

    assert mesh.nodes.numbers.dtype == np.uint32
    assert [g.topology for g in mesh.elements] == [
        Topology.C3D4,
        Topology.C3D5,
    ]


def test_mesh_from_container_accepts_empty_mesh() -> None:
    mesh = Mesh.from_container(
        {
            "nodes": {
                "numbers": [],
                "coordinates": np.array(()).reshape((0, DIM)),
            },
            "elements": {},
        },
    )

    assert mesh.nodes.coordinates.shape == (0, DIM)
    assert mesh.elements == ()


def test_mesh_from_container_accepts_h5py_group() -> None:
    with h5py.File.in_memory() as mesh_group:
        nodes_group = mesh_group.create_group("nodes")
        nodes_group.create_dataset("numbers", data=[1, 2, 3, 4])
        nodes_group.create_dataset("coordinates", data=np.zeros((4, DIM)))

        elements_group = mesh_group.create_group("elements")
        c3d4_group = elements_group.create_group("C3D4")
        c3d4_group.create_dataset("numbers", data=[10])
        c3d4_group.create_dataset("incidences", data=[[1, 2, 3, 4]])
        c3d4_group.create_dataset("nodes", data=[1, 2, 3, 4])
        mesh = Mesh.from_container(cast("Any", mesh_group))

    assert mesh.nodes.coordinates.shape == (4, DIM)
    assert len(mesh.elements) == 1
    assert mesh.elements[0].topology is Topology.C3D4


def test_mesh_from_container_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="is not a Mapping"):
        Mesh.from_container(cast("Any", object()))


def test_mesh_from_container_rejects_missing_nodes() -> None:
    with pytest.raises(ValueError, match="missing key: 'nodes'"):
        Mesh.from_container({"elements": {}})


def test_mesh_from_container_rejects_missing_elements() -> None:
    with pytest.raises(ValueError, match="missing key: 'elements'"):
        Mesh.from_container({"nodes": _nodes_container()})


def test_mesh_from_container_rejects_nodes_dataset() -> None:
    match = r"container\['nodes'\] must be a group"
    with pytest.raises(TypeError, match=match):
        Mesh.from_container({"nodes": [], "elements": {}})


def test_mesh_from_container_rejects_elements_dataset() -> None:
    match = r"container\['elements'\] must be a group"
    with pytest.raises(TypeError, match=match):
        Mesh.from_container({"nodes": _nodes_container(), "elements": []})


def test_mesh_from_container_rejects_element_dataset() -> None:
    with pytest.raises(TypeError, match="is not a Mapping"):
        Mesh.from_container(
            {"nodes": _nodes_container(), "elements": {"C3D4": []}},
        )


def test_mesh_from_container_rejects_unknown_node_references() -> None:
    match = r"Elements \(C3D4\) reference unknown nodes"
    with pytest.raises(ValueError, match=match):
        Mesh.from_container(
            {
                "nodes": {
                    "numbers": [1, 2, 3],
                    "coordinates": np.zeros((3, DIM)),
                },
                "elements": {"C3D4": _c3d4_container()},
            },
        )


def test_mesh_from_container_rejects_duplicate_element_numbers() -> None:
    c3d4 = _c3d4_container()
    c3d5 = _c3d5_container()
    c3d5["numbers"] = [10]

    with pytest.raises(ValueError, match="Element numbers are not unique"):
        Mesh.from_container(
            {
                "nodes": _nodes_container(),
                "elements": {"C3D4": c3d4, "C3D5": c3d5},
            },
        )


def test_named_region_from_container_converts_valid_mapping() -> None:
    region = NamedRegion.from_container(
        {
            "sets/element": {"fixed": [10, 20]},
            "sets/node": {"boundary": [1, 2, 3]},
        },
    )

    assert region.sets_element[0].name == "fixed"
    np.testing.assert_array_equal(region.sets_element[0].members, [10, 20])
    assert region.sets_node[0].name == "boundary"
    np.testing.assert_array_equal(region.sets_node[0].members, [1, 2, 3])


def test_named_region_from_container_accepts_empty_region() -> None:
    region = NamedRegion.from_container(
        {
            "sets/element": {},
            "sets/node": {},
        },
    )

    assert region.sets_element == ()
    assert region.sets_node == ()


def test_named_region_from_container_accepts_h5py_group() -> None:
    with h5py.File.in_memory() as root_group:
        sets_group = root_group.create_group("sets")
        element_group = sets_group.create_group("element")
        element_group.create_dataset("fixed", data=[10, 20])
        node_group = sets_group.create_group("node")
        node_group.create_dataset("boundary", data=[1, 2, 3])

        region = NamedRegion.from_container(cast("Any", root_group))

    assert region.sets_element[0].name == "fixed"
    np.testing.assert_array_equal(region.sets_element[0].members, [10, 20])
    assert region.sets_node[0].name == "boundary"
    np.testing.assert_array_equal(region.sets_node[0].members, [1, 2, 3])


def test_named_region_from_container_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="is not a Mapping"):
        NamedRegion.from_container(cast("Any", object()))


def test_named_region_from_container_rejects_missing_element_sets() -> None:
    with pytest.raises(ValueError, match="missing key: 'sets/element'"):
        NamedRegion.from_container({"sets/node": {}})


def test_named_region_from_container_rejects_missing_node_sets() -> None:
    with pytest.raises(ValueError, match="missing key: 'sets/node'"):
        NamedRegion.from_container({"sets/element": {}})


def test_named_region_from_container_rejects_element_sets_dataset() -> None:
    match = r"container\['sets/element'\] must be a group"
    with pytest.raises(TypeError, match=match):
        NamedRegion.from_container({"sets/element": [], "sets/node": {}})


def test_named_region_from_container_rejects_node_sets_dataset() -> None:
    match = r"container\['sets/node'\] must be a group"
    with pytest.raises(TypeError, match=match):
        NamedRegion.from_container({"sets/element": {}, "sets/node": []})


def test_named_region_from_container_rejects_bad_set_dataset() -> None:
    match = r"container\['fixed'\].*numeric array-like dataset"
    with pytest.raises(TypeError, match=match):
        NamedRegion.from_container(
            {
                "sets/element": {"fixed": ["invalid"]},
                "sets/node": {},
            },
        )


def test_named_region_from_container_rejects_invalid_set_members() -> None:
    match = r"Set\.members must be strongly sorted"
    with pytest.raises(ValueError, match=match):
        NamedRegion.from_container(
            {
                "sets/element": {"fixed": [20, 10]},
                "sets/node": {},
            },
        )
