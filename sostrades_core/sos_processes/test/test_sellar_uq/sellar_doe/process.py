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

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder
from sostrades_core.sos_processes.test.test_sellar_uq.sellar_doe.usecase import Study

if TYPE_CHECKING:
    from sostrades_core.execution_engine.sos_builder import SoSBuilder


class ProcessBuilder(BaseProcessBuilder):
    """A class that builds the SoSTrades process for the Sellar problem."""

    # Ontology information
    _ontology_data: ClassVar = {
        "label": "Sellar DOE Process",
        "description": "DOE on the Sellar problem",
        "type": "Fake",
        "source": "Vincent Drouet",
        "version": "",
    }

    def get_builders(self) -> list[SoSBuilder]:
        """Create the builders for the process."""
        disc_dir = "sostrades_core.sos_wrapping.test_discs.test_sellar_uq"
        mods_dict = {
            disc: f"{disc_dir}.{disc.lower()}.Sellar{disc}" for disc in ["Disc1", "Disc2", "Obj", "Cstr1", "Cstr2"]
        }
        builders_list = self.create_builder_list(
            mods_dict,
            ns_dict={
                "ns_sellar": f"{self.ee.study_name}",
            },
        )

        # Coupling
        coupling_builder = self.ee.factory.create_builder_coupling(Study.COUPLING_NAME)
        coupling_builder.set_builder_info("cls_builder", builders_list)
        self.ee.ns_manager.add_ns("ns_sellar", f"{self.ee.study_name}.{Study.COUPLING_NAME}")

        # Driver builder
        return self.ee.factory.create_mono_instance_driver("Eval", coupling_builder)
