# Copyright (c) 2023 Capgemini Engineering
# All rights reserved.
#
# Created on 09/07/2024, 15:26
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author:  Vincent Drouet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The main process of the Sellar problem."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder
from sostrades_core.sos_processes.test.test_sellar_uq.sellar_mdo.usecase import Study

if TYPE_CHECKING:
    from sostrades_core.execution_engine.sos_builder import SoSBuilder


class ProcessBuilder(BaseProcessBuilder):
    """A class that builds the SoSTrades process for the Sellar problem."""

    # Ontology information
    _ontology_data: ClassVar = {
        "label": "Sellar Process",
        "description": "Toy case with the Sellar problem",
        "type": "Fake",
        "source": "Vincent Drouet",
        "version": "",
    }

    def get_builders(self) -> list[SoSBuilder]:
        """Create the builders for the process."""
        disc_dir = "sellar_sostrades.disciplines"
        mods_dict = {
            disc: f"{disc_dir}.{disc.lower()}.Sellar{disc}" for disc in ["Disc1", "Disc2", "Obj", "Cstr1", "Cstr2"]
        }
        builders_list = self.create_builder_list(
            mods_dict,
            ns_dict={
                "ns_sellar": f"{self.ee.study_name}.{Study.SCENARIO_NAME}",
            },
        )

        # Coupling
        coupling_builder = self.ee.factory.create_builder_coupling(Study.COUPLING_NAME)
        coupling_builder.set_builder_info("cls_builder", builders_list)

        # Optim
        return self.ee.factory.create_optim_builder(Study.SCENARIO_NAME, [coupling_builder])
