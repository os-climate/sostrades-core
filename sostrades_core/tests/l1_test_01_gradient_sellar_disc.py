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
import pandas as pd
import numpy as np
from os.path import dirname
from itertools import product

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
from sostrades_core.execution_engine.design_var.design_var_disc import DesignVarDiscipline


class GradientSellar(AbstractJacobianUnittest):
    """
    Sellar gradients test class
    """
    # AbstractJacobianUnittest.DUMP_JACOBIAN = True
    np.random.seed(42)

    def analytic_grad_entry(self):
        return [
            self.test_01_analytic_gradient_default_dataframe_fill(),
        ]

    def setUp(self):
        """Initialize"""
        self.study_name = 'Test'
        self.ns = f'{self.study_name}'

    def tearDown(self):
        pass

    def test_01_analytic_gradient_default_dataframe_fill(self):
        """Test gradient for Sellar1 """

        # create exec engine and use discipline
        self.ee = ExecutionEngine(self.study_name)
        factory = self.ee.factory
        SellarDisc1Path = 'sostrades_core.sos_wrapping.test_discs.sellar_for_design_var.Sellar1'
        sellar1_disc = factory.get_builder_from_module('Sellar1', SellarDisc1Path)
        self.ee.ns_manager.add_ns_def({'ns_OptimSellar': self.ns})
        self.ee.factory.set_builders_to_coupling_builder(sellar1_disc)
        self.ee.configure()

        # update dictionary with values
        values_dict = {f'{self.ns}.x': pd.DataFrame(data={'index': [0, 1, 2, 3], 'value': [1., 1., 1., 1.]}),
                       f'{self.ns}.z': np.array([5., 2.]), f'{self.ns}.y_2': 12.058488150611574}

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.configure()

        self.ee.update_from_dm()
        self.ee.prepare_execution()
        disc = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_sellar_1.pkl',
                            discipline=disc,
                            step=1e-16,
                            derr_approx='complex_step',
                            threshold=1e-5,
                            local_data=values_dict,
                            inputs=[f'{self.ns}.x', f'{self.ns}.z', f'{self.ns}.y_2'],
                            outputs=[f'{self.ns}.y_1']
                            )

    def test_02_analytic_gradient_sellar_2(self):
        """Test gradient for Sellar2 """
        # create exec engine and use discipline

        self.ee = ExecutionEngine(self.study_name)
        factory = self.ee.factory
        SellarDisc1Path = 'sostrades_core.sos_wrapping.test_discs.sellar_for_design_var.Sellar2'
        sellar1_disc = factory.get_builder_from_module('Sellar2', SellarDisc1Path)
        self.ee.ns_manager.add_ns_def({'ns_OptimSellar': self.ns})
        self.ee.factory.set_builders_to_coupling_builder(sellar1_disc)
        self.ee.configure()

        values_dict = {f'{self.ns}.z': np.array([5., 2.]), f'{self.ns}.y_1': 25.588302369877685}

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.configure()
        self.ee.update_from_dm()
        self.ee.prepare_execution()
        disc = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_sellar_2.pkl',
                            discipline=disc,
                            step=1e-16,
                            derr_approx='complex_step',
                            threshold=1e-5,
                            local_data=values_dict,
                            inputs=[f'{self.ns}.z', f'{self.ns}.y_1'],
                            outputs=[f'{self.ns}.y_2']
                            )

    def test_03_analytic_gradient_sellar_problem(self):
        """Test gradient for SellarProblem """

        # create exec engine and use discipline

        self.ee = ExecutionEngine(self.study_name)
        factory = self.ee.factory
        SellarProblemPath = 'sostrades_core.sos_wrapping.test_discs.sellar_for_design_var.SellarProblem'
        sellar_problem = factory.get_builder_from_module('SellarProblem', SellarProblemPath)
        self.ee.ns_manager.add_ns_def({'ns_OptimSellar': self.ns})
        self.ee.factory.set_builders_to_coupling_builder(sellar_problem)
        self.ee.configure()

        values_dict = {f'{self.ns}.x': pd.DataFrame(data={'index': [0, 1, 2, 3], 'value': [1., 1., 1., 1.]}),
                       f'{self.ns}.z': np.array([5., 2.]), f'{self.ns}.y_1': 25.588302369877685,
                       f'{self.ns}.y_2': 12.058488150611574, f'{self.ns}.SellarProblem.local_dv': 10.0}

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.configure()
        self.ee.update_from_dm()
        self.ee.prepare_execution()
        disc = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_sellar_problem.pkl',
                            discipline=disc,
                            step=1e-16,
                            derr_approx='complex_step',
                            threshold=1e-5,
                            local_data=values_dict,
                            inputs=[f'{self.ns}.z', f'{self.ns}.y_1', f'{self.ns}.y_2', f'{self.ns}.x'],
                            outputs=[f'{self.ns}.obj', f'{self.ns}.c_1', f'{self.ns}.c_1']
                            )
