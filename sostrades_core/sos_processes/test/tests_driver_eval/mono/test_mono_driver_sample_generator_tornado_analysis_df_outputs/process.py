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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        "label": "Core Test Disc1 Tornado Chart",
        "description": "",
        "category": "",
        "version": "",
    }

    def get_builders(self):
        disc_dir = "sostrades_core.sos_wrapping.test_discs."
        mods_dict = {"DiscAllTypes": disc_dir + "disc_all_types.DiscAllTypes"}
        builder_list = self.create_builder_list(
            mods_dict,
            ns_dict={"ns_test": f'{self.ee.study_name}.Coupling.DiscAllTypes'},
        )
        coupling_builder = self.ee.factory.create_builder_coupling("Coupling")
        coupling_builder.set_builder_info("cls_builder", builder_list)
        eval_builder = self.ee.factory.create_mono_instance_driver("Eval", coupling_builder)

        return eval_builder
