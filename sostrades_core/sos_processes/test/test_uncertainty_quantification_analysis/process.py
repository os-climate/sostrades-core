'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/13-2023/11/03 Copyright 2023 Capgemini.

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

# -- Generate test 1 process
from __future__ import annotations

from typing import ClassVar

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    """Process for the UQ analysis."""

    # ontology information
    _ontology_data: ClassVar = {
        "label": "sos_trades_core.sos_processes.test.test_uncertainty_quantification",
        "description": "",
        "category": "",
        "version": "",
    }

    def get_builders(self):  # noqa: D102
        uq_name = "UncertaintyQuantification"
        ns_value = f"{self.ee.study_name}.{uq_name}"
        ns_dict = {"ns_sample_generator": ns_value, "ns_evaluator": ns_value, "ns_uncertainty_quantification": ns_value}
        self.ee.ns_manager.add_ns_def(ns_dict)

        builder = self.ee.factory.add_uq_builder(uq_name)

        return [builder]
