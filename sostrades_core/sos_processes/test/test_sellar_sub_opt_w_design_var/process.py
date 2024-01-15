'''
Copyright 2023 Capgemini

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
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Generate an optimization scenario
"""
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Sellar SubOpt process with Design Var and Func Manager',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        default initialisation test
        '''
        # add disciplines Sellar
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.sellar_for_design_var.'

        mod_func = 'sostrades_core.execution_engine.func_manager.func_manager_disc.FunctionManagerDisc'
        mod_dv = 'sostrades_core.execution_engine.design_var.design_var_disc.DesignVarDiscipline'

        mods_dict = {
                     'Sellar_Problem': disc_dir + 'SellarProblem',
                     'Sellar_2': disc_dir + 'Sellar2',
                     'Sellar_1': disc_dir + 'Sellar1',

                     }

        ns_dict = {'ns_optim': self.ee.study_name + '.Sellar.SellarOptimScenario',
                   'ns_OptimSellar': self.ee.study_name + '.Sellar.SellarOptimScenario',
                   'ns_functions': self.ee.study_name + '.Sellar.SellarOptimScenario'
                   }

        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)

        # coupling
        coupling_builder = self.ee.factory.create_builder_coupling(
            "SellarCoupling")
        coupling_builder.set_builder_info('cls_builder', builder_list)

        mods_dict_bis = {'FunctionManager': mod_func,
                         'DesignVar': mod_dv}

        builder_fm_ds_list = self.create_builder_list(mods_dict_bis, ns_dict=ns_dict)
        builder_fm_ds_list.append(coupling_builder)
        coupling_builders = self.ee.factory.create_builder_coupling(
            "Sellar")
        coupling_builders.set_builder_info('cls_builder', builder_fm_ds_list)
        return coupling_builders
