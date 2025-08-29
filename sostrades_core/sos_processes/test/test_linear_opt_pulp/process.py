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

"""
Generate a linear programming optimization scenario using PuLP algorithms
"""


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Linear Programming Optimization with PuLP',
        'description': 'Test process for linear programming optimization using PuLP algorithms',
        'category': 'Test',
        'version': '1.0',
    }

    def get_builders(self):
        '''
        Create optimization process with linear problem discipline
        '''
        # Define discipline module path
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.'

        # Modules dictionary - just the linear problem discipline
        mods_dict = {
            'LinearProblem': disc_dir + 'linear_problem.LinearProblem',
        }

        # Namespace dictionary
        ns_dict = {
            'ns_linear_optim': self.ee.study_name + '.LinearOptimScenario.LinearCoupling',
            'ns_OptimLinear': self.ee.study_name + '.LinearOptimScenario.LinearCoupling',
        }

        # Create builder list for the discipline
        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)

        # Create coupling builder
        coupling_builder = self.ee.factory.create_builder_coupling("LinearCoupling")
        coupling_builder.set_builder_info('cls_builder', builder_list)

        # Create optimization builder
        opt_builder = self.ee.factory.create_optim_builder(
            'LinearOptimScenario', [coupling_builder])

        return opt_builder
