'''
Copyright 2024 Capgemini.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import annotations

import shutil
import time
from pathlib import Path


def rmtree_safe(path: Path | str, retry: int = 100) -> None:
    """
    Remove a directory tree, and wait for confirmation.

    Useful to fix some issues with shutil.rmtree with network drives.

    Args:
        path: The directory tree to remove..
        retry: The maximum number of loop iterations to check if the directory is removed.

    Raises:
        RuntimeError: If the removal fails after the maximum number of retries.

    """
    path = Path(path).resolve()
    shutil.rmtree(path)

    attempts = 0
    while path.exists() and attempts < retry:
        time.sleep(0.05)
        attempts += 1

    if path.exists():
        msg = f"Failed to remove the directory {path} after {retry} retries."
        raise RuntimeError(msg)


def makedirs_safe(name: str | Path, mode: int = 511, exist_ok: bool = False, retry: int = 100) -> None:
    """
    Make a directory, but wait until it is really created.

    Useful to fix some issues with Path.makedirs with network drives.

    Args:
        name: The directory tree to create.
        mode: The mode of directory creation.
        exist_ok: Whether to pass if the directory already exists. If False, will raise a FileExistsError.
        retry: The maximum number of retries for creating the directory.

    Raises:
        RuntimeError: If the directory creation fails after the maximum number of retries.

    """
    dir_path = Path(name).resolve()
    dir_path.mkdir(mode=mode, exist_ok=exist_ok, parents=True)

    attempts = 0
    while not dir_path.exists() and attempts < retry:
        time.sleep(0.05)
        attempts += 1

    if not dir_path.exists():
        msg = f"Failed to create the directory {name} after {retry} retries."
        raise RuntimeError(msg)
