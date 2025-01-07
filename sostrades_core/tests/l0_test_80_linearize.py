'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/30-2024/06/10 Copyright 2023 Capgemini

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
import unittest
import warnings
from pathlib import Path

from gemseo.core.discipline.discipline import Discipline
from numpy import ComplexWarning

from sostrades_core.execution_engine.execution_engine import ExecutionEngine

# IMPORT USECASES
from sostrades_core.sos_processes.test.test_disc1_all_types.usecase import (
    Study as Study_disc1_all_types,
)
from sostrades_core.sos_processes.test.test_sellar_coupling.usecase import (
    Study as Study_sellar_coupling,
)
from sostrades_core.sos_processes.test.test_sellar_coupling_new_types._usecase import (
    Study as Study_sellar_coupling_new_types,
)
from sostrades_core.tools.folder_operations import rmtree_safe


class TestAnalyticGradients(unittest.TestCase):
    """
    Class to test analytic gradients of Sellar optim case
    """

    def setUp(self):
        self.dirs_to_del = []
        self.study_name = 'usecase'
        self.ns = f'{self.study_name}'
        self.repo = 'sostrades_core.sos_processes.test'
        warnings.filterwarnings('error', category=ComplexWarning)

    def tearDown(self):
        for dir_to_del in self.dirs_to_del:
            if Path(dir_to_del).is_dir():
                rmtree_safe(dir_to_del)

    def test_01_linearize_on_simple_disc(self):

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = 'test_disc1_all_types'

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        values_dict = Study_disc1_all_types.setup_usecase(self)

        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.prepare_execution()
        exec_eng.display_treeview_nodes()
        for proxy_disc in exec_eng.root_process.proxy_disciplines:
            mdo_disc = proxy_disc.discipline_wrapp.discipline
            mdo_disc.linearization_mode = Discipline.LinearizationMode.FINITE_DIFFERENCES
            mdo_disc.add_differentiated_inputs()
            mdo_disc.add_differentiated_outputs()
            mdo_disc.linearize(values_dict)
            print('LINEARIZE performed for ', proxy_disc.get_disc_full_name())
        mda_chain = exec_eng.root_process.discipline_wrapp.discipline
        mda_chain.linearization_mode = Discipline.LinearizationMode.FINITE_DIFFERENCES
        mda_chain.add_differentiated_inputs()
        mda_chain.add_differentiated_outputs()
        mda_chain.linearize(values_dict)
        print('LINEARIZE performed for root coupling')

    def test_02_linearize_on_sellar_coupling(self):

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = 'test_sellar_coupling'

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        values_dict = Study_sellar_coupling.setup_usecase(self)[0]

        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.prepare_execution()
        exec_eng.display_treeview_nodes()
        for proxy_disc in exec_eng.root_process.proxy_disciplines:
            mdo_disc = proxy_disc.discipline_wrapp.discipline
            mdo_disc.linearization_mode = Discipline.LinearizationMode.FINITE_DIFFERENCES
            mdo_disc.linearize(values_dict, compute_all_jacobians=True)
            print('LINEARIZE performed for ', proxy_disc.get_disc_full_name())
            added_values_dict = {key: exec_eng.dm.get_value(key) for key in
                                 proxy_disc.get_input_data_names(numerical_inputs=False) if key not in values_dict}
            values_dict.update(added_values_dict)
        exec_eng.root_process.discipline_wrapp.discipline.linearization_mode = Discipline.LinearizationMode.FINITE_DIFFERENCES
        exec_eng.root_process.discipline_wrapp.discipline.linearize(values_dict, compute_all_jacobians=True)
        print('LINEARIZE performed for ', exec_eng.root_process.get_disc_full_name())

    def test_03_linearize_on_sellar_coupling_new_types(self):

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = 'test_sellar_coupling_new_types'

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        values_dict = Study_sellar_coupling_new_types.setup_usecase(self)[0]

        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.prepare_execution()
        exec_eng.display_treeview_nodes()
        for proxy_disc in exec_eng.root_process.proxy_disciplines:
            mdo_disc = proxy_disc.discipline_wrapp.discipline
            mdo_disc.linearize(values_dict, compute_all_jacobians=True)
            print('LINEARIZE performed for ', proxy_disc.get_disc_full_name())

            added_values_dict = {key: exec_eng.dm.get_value(key) for key in
                                 proxy_disc.get_input_data_names(numerical_inputs=False) if key not in values_dict}
            values_dict.update(added_values_dict)
        exec_eng.root_process.discipline_wrapp.discipline.linearize(values_dict,
                                                                                                 compute_all_jacobians=True)
        print('LINEARIZE performed for ', exec_eng.root_process.get_disc_full_name())

    def test_04_check_jacobian_on_sellar_coupling(self):

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = 'test_sellar_coupling'

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        values_dict = Study_sellar_coupling.setup_usecase(self)[0]
        values_dict[f'{self.ns}.SellarCoupling.inner_mda_name'] = 'MDAGaussSeidel'

        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.prepare_execution()
        exec_eng.display_treeview_nodes()
        for proxy_disc in exec_eng.root_process.proxy_disciplines:
            mdo_disc = proxy_disc.discipline_wrapp.discipline
            assert (mdo_disc.check_jacobian(values_dict, derr_approx='complex_step',
                                            step=1e-15, threshold=1e-8, ))
            print('CHECK_JACOBIAN performed for ', proxy_disc.get_disc_full_name())

            added_values_dict = {key: exec_eng.dm.get_value(key) for key in
                                 proxy_disc.get_input_data_names(numerical_inputs=False) if key not in values_dict}
            values_dict.update(added_values_dict)
        assert (exec_eng.root_process.discipline_wrapp.discipline.check_jacobian(
            values_dict, linearization_mode='adjoint', derr_approx='complex_step', step=1e-15, threshold=1e-8, ))

        print('CHECK_JACOBIAN performed for ', exec_eng.root_process.get_disc_full_name())

    def test_05_check_jacobian_on_sellar_coupling_new_types(self):

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = 'test_sellar_coupling_new_types'

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        values_dict = Study_sellar_coupling_new_types.setup_usecase(self)[0]
        values_dict[f'{self.ns}.SellarCoupling.inner_mda_name'] = 'MDAGaussSeidel'
        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.prepare_execution()
        exec_eng.display_treeview_nodes()
        for proxy_disc in exec_eng.root_process.proxy_disciplines:
            mdo_disc = proxy_disc.discipline_wrapp.discipline
            assert (mdo_disc.check_jacobian(values_dict, derr_approx='complex_step',
                                            step=1e-15, threshold=1e-8, ))
            print('CHECK_JACOBIAN performed for ', proxy_disc.get_disc_full_name())

            added_values_dict = {key: exec_eng.dm.get_value(key) for key in
                                 proxy_disc.get_input_data_names(numerical_inputs=False) if key not in values_dict}
            values_dict.update(added_values_dict)

        assert (exec_eng.root_process.discipline_wrapp.discipline.check_jacobian(
            values_dict, derr_approx='complex_step', step=1e-15, threshold=1e-8, linearization_mode='adjoint'))
        print('CHECK_JACOBIAN performed for ', exec_eng.root_process.get_disc_full_name())

    def test_06_check_jacobian_specified_inputs_outputs(self):

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = 'test_sellar_coupling_new_types'

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        values_dict = Study_sellar_coupling_new_types.setup_usecase(self)[0]

        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.prepare_execution()
        exec_eng.display_treeview_nodes()
        mdo_disc = exec_eng.dm.get_disciplines_with_name('usecase.SellarCoupling.Sellar_2')[
            0].discipline_wrapp.discipline
        inputs = ['usecase.SellarCoupling.z']
        outputs = ['usecase.SellarCoupling.y_2']
        assert (mdo_disc.check_jacobian(values_dict, derr_approx='complex_step',
                                        step=1e-15, threshold=1e-8, inputs=inputs, outputs=outputs,
                                        output_column='value'))

    def test_07_linearize_on_simple_disc_user(self):

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = 'test_disc1_all_types'

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        values_dict = Study_disc1_all_types.setup_usecase(self)

        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.prepare_execution()
        exec_eng.display_treeview_nodes()
        for proxy_disc in exec_eng.root_process.proxy_disciplines:
            mdo_disc = proxy_disc.discipline_wrapp.discipline
            mdo_disc.linearization_mode = Discipline.LinearizationMode.ADJOINT
            mdo_disc.add_differentiated_inputs()
            mdo_disc.add_differentiated_outputs()
            mdo_disc.linearize(values_dict)
            print('LINEARIZE performed for ', proxy_disc.get_disc_full_name())
        mda_chain = exec_eng.root_process.discipline_wrapp.discipline
        mda_chain.linearization_mode = Discipline.LinearizationMode.ADJOINT
        mda_chain.add_differentiated_inputs()
        mda_chain.add_differentiated_outputs()
        mda_chain.linearize(values_dict)
        print('LINEARIZE performed for root coupling')
