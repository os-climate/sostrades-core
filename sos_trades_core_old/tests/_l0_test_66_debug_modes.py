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
unit test for optimization scenario
"""

import unittest
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.sos_processes.test.test_sellar_opt_w_func_manager_faulty.usecase import Study as study_sellar_opt_faulty


class TestDebugModes(unittest.TestCase):
    """
    Debug mode test class
    """

    def setUp(self):
        self.study_name = 'optim'
        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_sellar_opt_w_func_manager_faulty'

    def test_00_debug_mode_null(self):
        '''
        Launch test_sellar_opt_w_func_manager_faulty without a debug mode on
        '''
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)

        exec_eng.configure()

        usecase = study_sellar_opt_faulty(execution_engine=exec_eng)
        usecase.study_name = self.study_name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()
        exec_eng.display_treeview_nodes()
        exec_eng.execute()

    def test_01_debug_mode_nan(self):
        '''
        Launch sellar opt with a NaN injected in discipline Sellar3
        Then test debug mode "nan" on discipline Sellar3
        '''
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)

        exec_eng.configure()

        usecase = study_sellar_opt_faulty(execution_engine=exec_eng)
        usecase.study_name = self.study_name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)
        # activate debug mode == nan to raise error
        values_dict[f'{usecase.study_name}.{usecase.optim_name}.{usecase.subcoupling_name}.Sellar_3.error_string'] = 'nan'
        values_dict[f'{usecase.study_name}.{usecase.optim_name}.{usecase.subcoupling_name}.Sellar_3.debug_mode'] = 'nan'
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()
        exec_eng.display_treeview_nodes()
        try:
            exec_eng.execute()
            raise Exception('Execution worked, and it should not have')
        except ValueError as ve:
            assert 'NaN values found in Sellar_3' == ve.args[0]
        except:
            raise Exception('Execution failed, and not for the good reason')

    def test_02_debug_mode_input_change(self):
        '''
        Launch sellar opt with a change of +0.5 on input "y_1" injected in discipline Sellar3
        Then test debug mode "input_change" on discipline Sellar3
        '''
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)

        exec_eng.configure()

        usecase = study_sellar_opt_faulty(execution_engine=exec_eng)
        usecase.study_name = self.study_name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)
        values_dict[
            f'{usecase.study_name}.{usecase.optim_name}.{usecase.subcoupling_name}.Sellar_3.error_string'] = 'input_change'
        values_dict[f'{usecase.study_name}.{usecase.optim_name}.{usecase.subcoupling_name}.Sellar_3.debug_mode'] = 'input_change'
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()
        exec_eng.display_treeview_nodes()
        try:
            exec_eng.execute()
            raise Exception('Execution worked, and it should not have')
        except ValueError as ve:
            assert "Mismatch in .optim.SellarOptimScenario.SellarCoupling.y_1.value: 27.8 and 28.3 don't match" in ve.args[0]
        except:
            raise Exception('Execution failed, and not for the good reason')

    def test_03_debug_mode_linearize_data_change(self):
        '''
        Launch sellar opt with an input change injected on "y_1" during the compute_jacobian in discipline Sellar3
        Then test debug mode "linearize data change" on discipline Sellar3
        '''
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)

        exec_eng.configure()

        usecase = study_sellar_opt_faulty(execution_engine=exec_eng)
        usecase.study_name = self.study_name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)
        values_dict[
            f'{usecase.study_name}.{usecase.optim_name}.{usecase.subcoupling_name}.Sellar_3.error_string'] = 'linearize_data_change'
        values_dict[
            f'{usecase.study_name}.{usecase.optim_name}.{usecase.subcoupling_name}.Sellar_3.debug_mode'] = 'linearize_data_change'
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()
        exec_eng.display_treeview_nodes()
        try:
            exec_eng.execute()
            raise Exception('Execution worked, and it should not have')
        except ValueError as ve:
            assert "Mismatch in .optim.SellarOptimScenario.SellarCoupling.y_1.value: 27.8 and 28.3 don't match" in ve.args[0]
        except:
            raise Exception('Execution failed, and not for the good reason')

    def test_04_debug_mode_min_max_grad(self):
        '''
        Launch sellar opt with an abnormaly high gradient "y_2 vs y_1" during the compute_jacobian in discipline Sellar3
        Then test debug mode "min_max_grad" on discipline Sellar3
        '''
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)

        exec_eng.configure()

        usecase = study_sellar_opt_faulty(execution_engine=exec_eng)
        usecase.study_name = self.study_name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)
        values_dict[
            f'{usecase.study_name}.{usecase.optim_name}.{usecase.subcoupling_name}.Sellar_3.error_string'] = 'min_max_grad'
        values_dict[
            f'{usecase.study_name}.{usecase.optim_name}.{usecase.subcoupling_name}.Sellar_3.debug_mode'] = 'min_max_grad'

        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()
        exec_eng.display_treeview_nodes()
        try:
            exec_eng.execute()
            raise Exception('Execution worked, and it should not have')
        except ValueError as ve:
            assert 'in discipline <Sellar_3> : dr<optim.SellarOptimScenario.SellarCoupling.y_2> / dr<optim.SellarOptimScenario.SellarCoupling.y_1>: maximum gradient value is <10000000000.0>' == ve.args[0]
        except:
            raise Exception('Execution failed, and not for the good reason')

    def test_05_debug_mode_min_max_coupling(self):
        '''
        Launch sellar opt with an abnormaly high gradient "y_2 vs y_1" during the compute_jacobian in discipline Sellar3
        Then test debug mode "min_max_coupling" on discipline Sellar3
        '''
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)

        exec_eng.configure()

        usecase = study_sellar_opt_faulty(execution_engine=exec_eng)
        usecase.study_name = self.study_name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)
        values_dict[
            f'{usecase.study_name}.{usecase.optim_name}.{usecase.subcoupling_name}.Sellar_3.error_string'] = 'min_max_couplings'
        values_dict[
            f'{usecase.study_name}.{usecase.optim_name}.{usecase.subcoupling_name}.Sellar_3.debug_mode'] = 'min_max_couplings'

        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()
        exec_eng.display_treeview_nodes()
        try:
            exec_eng.execute()
            pass
        except ValueError as ve:
            assert '<optim.SellarOptimScenario.SellarCoupling.y_2> has the minimum coupling value <[12.27257053+0.j]>'
        except:
            raise Exception('Execution failed, and not for the good reason')


if '__main__' == __name__:
    cls = TestDebugModes()
    cls.setUp()
    cls.test_00_debug_mode_null()
