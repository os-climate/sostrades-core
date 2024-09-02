'''
Copyright 2024 Capgemini

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

from typing import TYPE_CHECKING, ClassVar

from sostrades_core.sos_processes.test.test_sellar_uq.sellar_doe.process import ProcessBuilder as ProcessBuilderDOE
from sostrades_core.sos_processes.test.test_sellar_uq.sellar_uq.usecase import Study

if TYPE_CHECKING:
    from sostrades_core.execution_engine.sos_builder import SoSBuilder


class ProcessBuilder(ProcessBuilderDOE):
    """The process builder for UQ on Sellar."""

    # Ontology information
    _ontology_data: ClassVar = {
        "label": "Sellar UQ Process",
        "description": "UQ study of the Sellar problem",
        "type": "Fake",
        "source": "Vincent Drouet",
        "version": "",
    }

    def get_builders(self) -> list[SoSBuilder]:
        """Create the builders for the process.

        Returns:
            The list of builders.
        """
        # Build the DOE process
        builders = super().get_builders()
        builders.append(self.ee.factory.add_uq_builder(Study.UQ_NAME))

        ns_dict = {
            "ns_sample_generator": f"{self.ee.study_name}.{Study.SAMPLE_GENERATOR_NAME}",
            "ns_evaluator": f"{self.ee.study_name}.Eval",
            "ns_uncertainty_quantification": f"{self.ee.study_name}.{Study.UQ_NAME}",
        }
        self.ee.ns_manager.add_ns_def(ns_dict)

        return builders
