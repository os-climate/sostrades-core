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
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Generate an optimization scenario
"""
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test MDA Div Copy Func Manager Process',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):
        '''
        default initialisation test
        '''
        # add disciplines Sellar

        disc_dir = 'sos_trades_core.sos_wrapping.test_discs.mda_div_idf.'
        mod_func = 'sos_trades_core.execution_engine.func_manager.func_manager_disc.FunctionManagerDisc'

        mods_dict = {
            'SellarCopy': disc_dir + 'SellarCopy',
            'Sellar_Problem': disc_dir + 'SellarProblem',
            'FunctionManager': mod_func
        }
        # send only builders
        ns_dict = {'ns_OptimSellar': self.ee.study_name + '.SellarCoupling', 'ns_functions': self.ee.study_name +
                   '.' + 'SellarCoupling', 'ns_optim': self.ee.study_name + '.' + 'SellarCoupling'}
        list_builders = self.create_builder_list(mods_dict, ns_dict=ns_dict)

        builder_sellar = self.ee.factory.get_builder_from_process(
            'sos_trades_core.sos_processes.test', 'test_mda_div')
        m = [list_builders[0]]
        m.append(builder_sellar)
        m.append(list_builders[1])
        m.append(list_builders[2])
        """
        coupling_builder = self.ee.factory.create_builder_coupling("SellarCoupling")
        coupling_builder.set_builder_info('cls_builder', list_builders)
        coupling_builder.set_builder_info('with_data_io', True)
        """
        return m
