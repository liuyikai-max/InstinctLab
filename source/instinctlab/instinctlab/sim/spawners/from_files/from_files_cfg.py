# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Configuration for spawning assets from mesh files."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.sim import converters
from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg
from isaaclab.utils import configclass

from . import from_files


@configclass
class MeshFileCfg(converters.MeshConverterCfg, FileCfg):
    """Mesh file (OBJ, STL, FBX) to spawn asset from.

    It uses the :class:`MeshConverter` class to create a USD file from the mesh and spawns the
    imported USD file. Similar to the :class:`UsdFileCfg`, the generated USD file can be modified
    by specifying the respective properties in the configuration class.

    See :meth:`spawn_from_mesh` for more information.

    TODO: Typical ___FileCfg inherit FileCfg before converters.___ConverterCfg, which I don't think through here.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.
    """

    func: Callable = from_files.spawn_from_mesh
