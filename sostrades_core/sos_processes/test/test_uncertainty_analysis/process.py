'''
Copyright 2025 Capgemini

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

from typing import ClassVar

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):  # noqa: D101
    # Ontology information
    _ontology_data: ClassVar[dict[str, str]] = {
        "label": "Core test Sellar Monte Carlo sampling.",
        "description": "",
        "category": "",
        "version": "",
    }

    def get_builders(self):  # noqa: D102
        # Create the post-treatment discipline
        post_disc = "sostrades_core.sos_wrapping.analysis_discs.uncertainty_analysis.UncertaintyAnalysis"
        mods_dict = {"MC post": post_disc}
        builder_list = self.create_builder_list(
            mods_dict,
            ns_dict={
                "ns_driver_MC": f"{self.ee.study_name}.Eval_MC",
            },
        )

        return builder_list
