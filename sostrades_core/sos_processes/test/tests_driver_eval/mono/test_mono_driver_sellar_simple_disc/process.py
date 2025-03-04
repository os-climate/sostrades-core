'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/10-2023/11/03 Copyright 2023 Capgemini

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
        'label': 'Core Test Simple Sellar Mono instance Eval',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''Default initialisation test'''
        # add disciplines Sellar
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.'
        mods_dict = {
            'Sellar_Problem': disc_dir + 'sellar.SellarProblem',
            'Sellar_2': disc_dir + 'sellar.Sellar2',
            'Sellar_1': disc_dir + 'sellar.Sellar1',
        }
        builder_list_sellar = self.create_builder_list(
            mods_dict,
            ns_dict={
                'ns_OptimSellar': self.ee.study_name,
            },
        )
        eval_name = 'Eval'
        doe_eval_builder = self.ee.factory.create_mono_instance_driver(
            eval_name, builder_list_sellar
        )

        mods_dict2 = {'Simple_Disc': disc_dir + 'simple_disc.SimpleDisc'}
        builder_list2 = self.create_builder_list(
            mods_dict2, ns_dict={'ns_z': f'{self.ee.study_name}.Eval'}
        )

        builder_list2.append(doe_eval_builder)

        return builder_list2
