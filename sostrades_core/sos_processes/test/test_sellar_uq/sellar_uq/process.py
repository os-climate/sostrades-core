# Copyright (c) 2024 Capgemini Engineering
# All rights reserved.
#
# Created on 30/07/2024, 11:23
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author:  Vincent Drouet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The process for the UQ study on the Sellar MDA."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from sellar_sostrades.sos_processes.sellar_doe.process import ProcessBuilder as ProcessBuilderDOE
from sellar_sostrades.sos_processes.sellar_uq.usecase import Study

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
