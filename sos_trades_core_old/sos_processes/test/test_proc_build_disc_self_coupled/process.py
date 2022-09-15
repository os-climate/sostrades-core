'''
Copyright 2022 Airbus SAS

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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
# -- Generate test 1 process
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_processes.test.test_proc_build_disc_self_coupled',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        mod_path = 'sostrades_core.sos_wrapping.test_discs.disc_self_coupled.SelfCoupledDiscipline'

        disc_builder = self.ee.factory.get_builder_from_module(
            'SelfCoupledDiscipline', mod_path)

        return [disc_builder]