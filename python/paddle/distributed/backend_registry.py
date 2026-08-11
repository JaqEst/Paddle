# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Any, Callable

ProcessGroupBackendFactory = Callable[..., Any]

_BUILTIN_PROCESS_GROUP_BACKENDS = frozenset(
    ['nccl', 'gloo', 'heter', 'xccl', 'bkcl', 'flagcx']
)
_process_group_backend_registry: dict[str, ProcessGroupBackendFactory] = {}


def register_process_group_backend(
    backend: str, factory: ProcessGroupBackendFactory
) -> None:
    """
    Register a Python process group backend factory.

    The factory is called as::

        factory(
            store,
            rank,
            world_size,
            group_id,
            pg_options,
            group_name=group_name,
            nccl_comm_init_option=nccl_comm_init_option,
            nccl_config=nccl_config,
        )

    It should return an object compatible with Paddle's process group methods.
    """
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a non-empty string")
    if backend in _BUILTIN_PROCESS_GROUP_BACKENDS:
        raise ValueError(f"backend {backend} is built-in and cannot be registered")
    if not callable(factory):
        raise TypeError("factory must be callable")

    _process_group_backend_registry[backend] = factory


def unregister_process_group_backend(backend: str) -> None:
    _process_group_backend_registry.pop(backend, None)


def is_process_group_backend(backend: str) -> bool:
    return (
        backend in _BUILTIN_PROCESS_GROUP_BACKENDS
        or backend in _process_group_backend_registry
    )


def get_process_group_backend_factory(
    backend: str,
) -> ProcessGroupBackendFactory | None:
    return _process_group_backend_registry.get(backend)
