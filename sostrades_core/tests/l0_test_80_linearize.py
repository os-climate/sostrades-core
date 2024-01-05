'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/30-2023/11/03 Copyright 2023 Capgemini

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
import warnings

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
"""

import unittest
from time import sleep
from shutil import rmtree
from pathlib import Path

from sostrades_core.execution_engine.execution_engine import ExecutionEngine

# IMPORT USECASES
from sostrades_core.sos_processes.test.test_disc1_all_types.usecase import Study as Study_disc1_all_types
from sostrades_core.sos_processes.test.test_sellar_coupling.usecase import Study as Study_sellar_coupling
from sostrades_core.sos_processes.test.test_sellar_coupling_new_types._usecase import \
    Study as Study_sellar_coupling_new_types
from numpy import ComplexWarning


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
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

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
            mdo_disc = proxy_disc.mdo_discipline_wrapp.mdo_discipline
            mdo_disc.linearize(values_dict)
            print('LINEARIZE performed for ', proxy_disc.get_disc_full_name())
        exec_eng.root_process.mdo_discipline_wrapp.mdo_discipline.linearize(values_dict)
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
        for proxy_disc in exec_eng.root_process.proxy_disciplines[0].proxy_disciplines:
            mdo_disc = proxy_disc.mdo_discipline_wrapp.mdo_discipline
            mdo_disc.linearize(values_dict)
            print('LINEARIZE performed for ', proxy_disc.get_disc_full_name())
        exec_eng.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.linearize(values_dict)
        print('LINEARIZE performed for ', exec_eng.root_process.proxy_disciplines[0].get_disc_full_name())

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
        for proxy_disc in exec_eng.root_process.proxy_disciplines[0].proxy_disciplines:
            mdo_disc = proxy_disc.mdo_discipline_wrapp.mdo_discipline
            mdo_disc.linearize(values_dict)
            print('LINEARIZE performed for ', proxy_disc.get_disc_full_name())
        exec_eng.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.linearize(values_dict)
        print('LINEARIZE performed for ', exec_eng.root_process.proxy_disciplines[0].get_disc_full_name())

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
        for proxy_disc in exec_eng.root_process.proxy_disciplines[0].proxy_disciplines:
            mdo_disc = proxy_disc.mdo_discipline_wrapp.mdo_discipline
            assert (mdo_disc.check_jacobian(values_dict, derr_approx='complex_step',
                                            step=1e-15, threshold=1e-8, ))
            print('CHECK_JACOBIAN performed for ', proxy_disc.get_disc_full_name())
        assert (exec_eng.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.check_jacobian(
            values_dict, linearization_mode='adjoint', derr_approx='complex_step', step=1e-15, threshold=1e-8, ))
        print('CHECK_JACOBIAN performed for ', exec_eng.root_process.proxy_disciplines[0].get_disc_full_name())

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
        for proxy_disc in exec_eng.root_process.proxy_disciplines[0].proxy_disciplines:
            mdo_disc = proxy_disc.mdo_discipline_wrapp.mdo_discipline
            assert (mdo_disc.check_jacobian(values_dict, derr_approx='complex_step',
                                            step=1e-15, threshold=1e-8, ))
            print('CHECK_JACOBIAN performed for ', proxy_disc.get_disc_full_name())
        assert (exec_eng.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.check_jacobian(
            values_dict, derr_approx='complex_step', step=1e-15, threshold=1e-8, linearization_mode='adjoint'))
        print('CHECK_JACOBIAN performed for ', exec_eng.root_process.proxy_disciplines[0].get_disc_full_name())

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
            0].mdo_discipline_wrapp.mdo_discipline
        inputs = ['usecase.SellarCoupling.z']
        outputs = ['usecase.SellarCoupling.y_2']
        assert (mdo_disc.check_jacobian(values_dict, derr_approx='complex_step',
                                        step=1e-15, threshold=1e-8, inputs=inputs, outputs=outputs,
                                        output_column='value'))