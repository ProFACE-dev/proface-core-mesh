# SPDX-FileCopyrightText: 2026 Stefano Miccoli <stefano.miccoli@polimi.it>
# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

from collections.abc import Mapping
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt

type Group = Mapping[str, Any]


#
# attrs helpers
#
cmp_numpy = attrs.cmp_using(eq=np.array_equal)


#
# helper functions for reading h5py groups and datasets
#


def _get(container: Group, key: str) -> Any:
    try:
        return container[key]
    except KeyError as exc:
        msg = f"mesh container missing key: {key!r}"
        raise ValueError(msg) from exc


def array[T: np.number](
    container: Group, key: str, *, dtype: type[T]
) -> npt.NDArray[T]:
    value = _get(container, key)
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        msg = f"mesh container[{key!r}] must be a numeric array-like dataset"
        raise TypeError(msg) from exc


def group(container: Group, key: str) -> Group:
    value = _get(container, key)
    if not isinstance(value, Mapping):
        msg = f"mesh container[{key!r}] must be a group"
        raise TypeError(msg)
    return value
