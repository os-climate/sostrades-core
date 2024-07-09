'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2023/11/03 Copyright 2023 Capgemini

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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sostrades_core.execution_engine.execution_engine import ExecutionEngine
    from sostrades_core.execution_engine.sos_builder import SoSBuilder


class BaseProcessBuilder:
    """Generic class to inherit to build processes."""

    def __init__(self, ee: ExecutionEngine) -> None:
        self.ee = ee
        self.logger = ee.logger.getChild(self.__class__.__name__)

    def get_builders(self) -> list:
        return []

    def create_builder_list(
        self, mods_dict: dict[str, str], ns_dict: dict[str, str] = None, associate_namespace: bool = False
    ) -> list[SoSBuilder]:
        """Define a base namespace and instantiate builders iterating over a list of module paths.

        Args:
            mods_dict: The dictionary containing the module path for each discipline.
            ns_dict: The dictionary of namespaces.
            associate_namespace: Whether to replace existing namespaces.

        Returns:
            The list of discipline builders.
        """
        if associate_namespace:
            clean_existing = False
        else:
            clean_existing = True

        ns_ids = []
        if ns_dict is not None:
            ns_ids = self.ee.ns_manager.add_ns_def(ns_dict, clean_existing=clean_existing)
        builders = []

        for disc_name, mod_path in mods_dict.items():
            a_b = self.ee.factory.get_builder_from_module(disc_name, mod_path)
            if associate_namespace:
                a_b.associate_namespaces(ns_ids)
            builders.append(a_b)
        return builders
