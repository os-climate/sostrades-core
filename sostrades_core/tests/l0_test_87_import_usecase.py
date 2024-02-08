'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/05-2023/11/03 Copyright 2023 Capgemini

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
from logging import Handler

from pandas._testing import assert_frame_equal

from sostrades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for doe scenario
"""

import unittest
from numpy import array
import pandas as pd


from importlib import import_module


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestSoSimportUsecase(unittest.TestCase):
    # FIXME: flatten_subprocess => tests are passing but infinite configuration loop is present.
    def check_discipline_value(self, my_disc, my_data_name, target_value, print_flag=True, ioType='in'):
        my_data = my_disc.get_data_io_from_key(
            ioType, my_data_name)
        my_value = my_data['value']
        if isinstance(my_value, pd.DataFrame):
            assert_frame_equal(target_value, my_value)
        else:
            self.assertEqual(target_value, my_value)
        if print_flag:
            print(my_data_name + ': ', my_value)

    def check_discipline_values(self, my_disc, target_values_dict, print_flag=True, ioType='in'):
        if print_flag:
            print(
                f'Check_discipline value for {my_disc.get_disc_full_name()}:')
        for key in target_values_dict.keys():
            self.check_discipline_value(
                my_disc, key, target_value=target_values_dict[key], print_flag=print_flag, ioType=ioType)
        if print_flag:
            print('\n')

    def setUp(self):

        self.repo = 'sostrades_core.sos_processes.test'
        self.with_modal = True

    def test_1_usecase_import_multi_instances_eval_simple_disc1_disc3(self):
        """
        This test checks the usecase import capability in multi instance mode with eval
        It uses the test_disc1_disc3_list nested process 
        """
        from os.path import join, dirname
        ref_dir = join(dirname(__file__), 'data')
        join(ref_dir, 'dump_load_cache')

        # The generator eval process
        repo_name = self.repo + '.tests_driver_eval.multi'
        self.repo = self.repo + '.disc1_disc3'
        proc_name = 'test_multi_driver_subprocess_1_3'
        usecase_name = 'usecase_with_ref'

        # Creation of the study from the associated usecase
        self.study_name = usecase_name
        imported_module = import_module(
            '.'.join([repo_name, proc_name, usecase_name]))

        study_dump = imported_module.Study(run_usecase=True)

        study_dump.load_data()

        # study_dump.run() # In GUI it depends if we do run or not

        # Check the created study

        self.exec_eng = study_dump.ee

        self.exec_eng.display_treeview_nodes()
        ref_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.Eval.ReferenceScenario.Disc1')[0]

        # In the study creation it is provided x = 2.0
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # Load the anonymized dict from associated selected sub_process

        based_on_uc_name = True
        if based_on_uc_name:
            sub_process_name = 'test_disc1_disc3_list'
            sub_process_usecase_name = 'usecase'
            anonymize_input_dict_from_usecase = study_dump.static_load_raw_usecase_data(
                self.repo, sub_process_name, sub_process_usecase_name)
        else:
            anonymize_input_dict_from_usecase = {}
            anonymize_input_dict_from_usecase['<study_ph>.a'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.x'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.z'] = 1.2
            anonymize_input_dict_from_usecase['<study_ph>.Disc1.b'] = 4.0
            anonymize_input_dict_from_usecase['<study_ph>.Disc3.constant'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.Disc3.power'] = 2

        # Update the reference from the selected imported usecase anonymized
        # dict

        dict_values = {}
        if self.with_modal:
            process_builder_parameter_type = ProcessBuilderParameterType(
                self.repo, sub_process_name, sub_process_usecase_name)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            # process_builder_parameter_type.usecase_data = {}
            dict_values[
                f'{self.study_name}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            dict_values[f'{self.study_name}.Eval.usecase_data'] = anonymize_input_dict_from_usecase

        study_dump.load_data(from_input_dict=dict_values)

        # Check that the reference has been updated

        # In the anonymised dict of the selected usecase it is provided x = 3.0
        target_x = 3.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

    def test_2_usecase_import_multi_instances_eval_generator_cp_disc1_disc3(self):
        """
        This test checks the usecase import capability in mono instance mode with doe algo product generator + eval
        It uses the test_disc1_disc3_list nested process 
        """
        from os.path import join, dirname
        ref_dir = join(dirname(__file__), 'data')
        join(ref_dir, 'dump_load_cache')

        # The generator eval process
        repo_name = self.repo + '.tests_driver_eval.multi'
        self.repo = self.repo + '.disc1_disc3'
        proc_name = 'test_multi_driver_sample_generator_subprocess_1_3'
        usecase_name = 'usecase_with_ref'

        # Creation of the study from the associated usecase
        self.study_name = usecase_name
        imported_module = import_module(
            '.'.join([repo_name, proc_name, usecase_name]))

        study_dump = imported_module.Study(run_usecase=True)

        study_dump.load_data()

        # study_dump.run() # In GUI it depends if we do run or not

        # Check the created study
        self.exec_eng = study_dump.ee

        self.exec_eng.display_treeview_nodes(True)
        ref_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.Eval.ReferenceScenario.Disc1')[0]

        # In the study creation it is provided x = 2.0
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # Load the anonymized dict from associated selected sub_process

        based_on_uc_name = True
        if based_on_uc_name:
            sub_process_name = 'test_disc1_disc3_list'
            sub_process_usecase_name = 'usecase'
            anonymize_input_dict_from_usecase = study_dump.static_load_raw_usecase_data(
                self.repo, sub_process_name, sub_process_usecase_name)
        else:
            anonymize_input_dict_from_usecase = {}
            anonymize_input_dict_from_usecase['<study_ph>.a'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.x'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.z'] = 1.2
            anonymize_input_dict_from_usecase['<study_ph>.Disc1.b'] = 4.0
            anonymize_input_dict_from_usecase['<study_ph>.Disc3.constant'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.Disc3.power'] = 2

        # Update the reference from the selected imported usecase anonymized
        # dict

        dict_values = {}
        if self.with_modal:
            process_builder_parameter_type = ProcessBuilderParameterType(
                self.repo, sub_process_name, sub_process_usecase_name)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            # process_builder_parameter_type.usecase_data = {}
            dict_values[
                f'{self.study_name}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            dict_values[f'{self.study_name}.Eval.usecase_data'] = anonymize_input_dict_from_usecase

        study_dump.load_data(from_input_dict=dict_values)

        # Check that the reference has been updated

        # In the anonymised dict of the selected usecase it is provided x = 3.0
        target_x = 3.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

    def _test_3_usecase_import_mono_instances_eval_generator_doe_disc1_disc3(self):
        """
        This test checks the usecase import capability in mono instance mode with doe algo product generator + eval
        It uses the test_disc1_disc3_list nested process 
        """
        from os.path import join, dirname

        join(dirname(__file__), 'data')

        # The generator eval process
        repo_name = self.repo + '.tests_driver_eval.mono'
        self.repo = self.repo + '.disc1_disc3'

        proc_name = 'test_mono_driver_sample_generator_subprocess_1_3'
        usecase_name = 'usecase1_doe_mono'
        coupling_name = 'D1_D3_Coupling'

        # Creation of the study from the associated usecase
        self.study_name = usecase_name
        imported_module = import_module(
            '.'.join([repo_name, proc_name, usecase_name]))

        study_dump = imported_module.Study(run_usecase=True)

        study_dump.load_data()

        # study_dump.run() # In GUI it depends if we do run or not

        # Check the created study

        self.exec_eng = study_dump.ee

        self.exec_eng.display_treeview_nodes()
        ref_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.Eval.{coupling_name}.Disc1')[0]

        # In the study creation it is provided x = 2.0
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # Load the anonymized dict from associated selected sub_process

        based_on_uc_name = True
        if based_on_uc_name:
            sub_process_name = 'test_disc1_disc3_coupling'
            sub_process_usecase_name = 'usecase'
            anonymize_input_dict_from_usecase = study_dump.static_load_raw_usecase_data(
                self.repo, sub_process_name, sub_process_usecase_name)
        else:
            anonymize_input_dict_from_usecase = {}
            anonymize_input_dict_from_usecase[f'<study_ph>.{coupling_name}.a'] = 3.0
            anonymize_input_dict_from_usecase[f'<study_ph>.{coupling_name}.x'] = 3.0
            anonymize_input_dict_from_usecase[f'<study_ph>.{coupling_name}.z'] = 1.2
            anonymize_input_dict_from_usecase[f'<study_ph>.{coupling_name}.Disc1.b'] = 4.0
            anonymize_input_dict_from_usecase[f'<study_ph>.{coupling_name}.Disc3.constant'] = 3.0
            anonymize_input_dict_from_usecase[f'<study_ph>.{coupling_name}.Disc3.power'] = 2

        # Update the reference from the selected imported usecase anonymized
        # dict

        dict_values = {}
        if self.with_modal:
            process_builder_parameter_type = ProcessBuilderParameterType(
                self.repo, sub_process_name, sub_process_usecase_name)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            # process_builder_parameter_type.usecase_data = {}
            dict_values[
                f'{self.study_name}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            dict_values[f'{self.study_name}.Eval.usecase_data'] = anonymize_input_dict_from_usecase

        study_dump.load_data(from_input_dict=dict_values)

        # Check that the reference has been updated

        # In the anonymised dict of the selected usecase it is provided x = 3.0
        target_x = 3.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

    def test_4_usecase_import_multi_instances_eval_simple_sellar(self):
        """
        This test checks the usecase import capability in multi instance mode with eval 
        It uses the sellar_coupling nested process
        """
        from os.path import join, dirname
        ref_dir = join(dirname(__file__), 'data')
        join(ref_dir, 'dump_load_cache')

        # The generator eval process
        repo_name = self.repo + '.tests_driver_eval.multi'
        self.repo = self.repo + ".sellar"
        proc_name = 'test_multi_driver_sellar'

        usecase_name = 'usecase1_with_ref'

        # Associated nested subprocess
        sub_process_name = 'test_sellar_list'
        sub_process_usecase_name = 'usecase'

        # Creation of the study from the associated usecase
        self.study_name = usecase_name
        imported_module = import_module(
            '.'.join([repo_name, proc_name, usecase_name]))

        study_dump = imported_module.Study(run_usecase=True)

        study_dump.load_data()

        # study_dump.run() # I remove run to be as in GUI test

        # Check the created study

        self.exec_eng = study_dump.ee

        self.exec_eng.display_treeview_nodes()

        ref_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.Eval.ReferenceScenario.Sellar_1')[0]

        # In the study creation it is provided x = array([2.])
        target_x = array([2.])
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # Load the anonymized dict from associated selected sub_process

        based_on_uc_name = True
        if based_on_uc_name:
            anonymize_input_dict_from_usecase = study_dump.static_load_raw_usecase_data(
                self.repo, sub_process_name, sub_process_usecase_name)
        else:
            # Below was as it was done in the console first
            # ==================================================================
            # anonymize_input_dict_from_usecase = {}
            # anonymize_input_dict_from_usecase['<study_ph>.x'] = array([1.])
            # anonymize_input_dict_from_usecase['<study_ph>.y_1'] = array([1.])
            # anonymize_input_dict_from_usecase['<study_ph>.y_2'] = array([1.])
            # anonymize_input_dict_from_usecase['<study_ph>.z'] = array([1., 1.])
            # anonymize_input_dict_from_usecase['<study_ph>.Sellar_Problem.local_dv'] = 10.
            # ==================================================================

            # Here is as it has been modified to be as in the GUI and/or  csv
            # anonymised dict import.
            anonymize_input_dict_from_usecase = {}
            anonymize_input_dict_from_usecase['<study_ph>.x'] = [1.]
            anonymize_input_dict_from_usecase['<study_ph>.y_1'] = [1.]
            anonymize_input_dict_from_usecase['<study_ph>.y_2'] = [1.]
            anonymize_input_dict_from_usecase['<study_ph>.z'] = [1., 1.]
            anonymize_input_dict_from_usecase['<study_ph>.Sellar_Problem.local_dv'] = 10.

        # Update the reference from the selected imported usecase anonymised
        # dict

        dict_values = {}
        if self.with_modal:
            process_builder_parameter_type = ProcessBuilderParameterType(
                self.repo, sub_process_name, sub_process_usecase_name)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            # process_builder_parameter_type.usecase_data = {}
            dict_values[
                f'{self.study_name}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            dict_values[f'{self.study_name}.Eval.usecase_data'] = anonymize_input_dict_from_usecase
        study_dump.load_data(from_input_dict=dict_values)

        # Check that the reference has been updated

        # In the anonymised dict of the selected usecase it is provided x =
        # array([1.])
        target_x = array([1.])
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # study_dump.run()

    def test_5_usecase_import_multi_instances_eval_generator_cp_sellar(self):
        """
        This test checks the usecase import capability in multi instance mode with generator + eval
        It uses the sellar_coupling nested process
        """
        from os.path import join, dirname
        ref_dir = join(dirname(__file__), 'data')
        join(ref_dir, 'dump_load_cache')

        with_coupling = False  # In multi instances only False is of interest

        # The generator eval process
        repo_name = self.repo + '.tests_driver_eval.multi'
        self.repo = self.repo + '.sellar'
        if with_coupling:
            proc_name = 'test_multi_driver_sample_generator_sellar_coupling'
        else:
            proc_name = 'test_multi_driver_sample_generator_sellar_simple'

        usecase_name = 'usecase1_cp_multi_with_ref'

        # Associated nested subprocess
        if with_coupling:
            sub_process_name = 'test_sellar_coupling'
        else:
            sub_process_name = 'test_sellar_list'

        sub_process_usecase_name = 'usecase'

        # Creation of the study from the associated usecase
        self.study_name = usecase_name
        imported_module = import_module(
            '.'.join([repo_name, proc_name, usecase_name]))

        study_dump = imported_module.Study(run_usecase=True)

        study_dump.load_data()

        # study_dump.run() # I remove run to be as in GUI test

        # Check the created study

        self.exec_eng = study_dump.ee

        self.exec_eng.display_treeview_nodes()
        if with_coupling:
            ref_disc = self.exec_eng.dm.get_disciplines_with_name(
                f'{self.study_name}.Eval.ReferenceScenario.SellarCoupling.Sellar_1')[0]
        else:
            ref_disc = self.exec_eng.dm.get_disciplines_with_name(
                f'{self.study_name}.Eval.ReferenceScenario.Sellar_1')[0]

        # In the study creation it is provided x = array([2.])
        target_x = array([2.])
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # Load the anonymized dict from associated selected sub_process

        based_on_uc_name = True
        if based_on_uc_name:
            anonymize_input_dict_from_usecase = study_dump.static_load_raw_usecase_data(
                self.repo, sub_process_name, sub_process_usecase_name)
        else:
            if with_coupling:
                # Below was as it was done in the console first
                # ==============================================================
                # anonymize_input_dict_from_usecase = {}
                # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.x'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_1'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_2'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.z'] = array([1., 1.])
                # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.Sellar_Problem.local_dv'] = 10.
                # ==============================================================

                # Here is as it has been modified to be as in the GUI and/or
                # csv anonymised dict import.
                anonymize_input_dict_from_usecase = {}
                anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.x'] = [
                    1.]
                anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_1'] = [
                    1.]
                anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_2'] = [
                    1.]
                anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.z'] = [
                    1., 1.]
                anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.Sellar_Problem.local_dv'] = 10.

            else:
                # Below was as it was done in the console first
                # ==============================================================
                # anonymize_input_dict_from_usecase = {}
                # anonymize_input_dict_from_usecase['<study_ph>.x'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.y_1'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.y_2'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.z'] = array([1., 1.])
                # anonymize_input_dict_from_usecase['<study_ph>.Sellar_Problem.local_dv'] = 10.
                # ==============================================================

                # Here is as it has been modified to be as in the GUI and/or
                # csv anonymised dict import.
                anonymize_input_dict_from_usecase = {}
                anonymize_input_dict_from_usecase['<study_ph>.x'] = [1.]
                anonymize_input_dict_from_usecase['<study_ph>.y_1'] = [1.]
                anonymize_input_dict_from_usecase['<study_ph>.y_2'] = [1.]
                anonymize_input_dict_from_usecase['<study_ph>.z'] = [1., 1.]
                anonymize_input_dict_from_usecase['<study_ph>.Sellar_Problem.local_dv'] = 10

        # Update the reference from the selected imported usecase anonymised
        # dict

        dict_values = {}
        if self.with_modal:
            process_builder_parameter_type = ProcessBuilderParameterType(
                self.repo, sub_process_name, sub_process_usecase_name)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            # process_builder_parameter_type.usecase_data = {}
            dict_values[
                f'{self.study_name}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            dict_values[f'{self.study_name}.Eval.usecase_data'] = anonymize_input_dict_from_usecase
        study_dump.load_data(from_input_dict=dict_values)

        # Check that the reference has been updated

        # In the anonymised dict of the selected usecase it is provided x =
        # array([1.])
        target_x = array([1.])
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # study_dump.run()

    def _test_6_usecase_import_mono_instances_eval_generator_doe_sellar(self):
        """
        This test checks the usecase import capability in mono instance mode with generator  + eval
        It uses the sellar_coupling nested process
        """
        from os.path import join, dirname
        ref_dir = join(dirname(__file__), 'data')
        join(ref_dir, 'dump_load_cache')
        # TODO: ask Carlos
        with_coupling = True  # In mono instance only True is of interest

        # The generator eval process
        repo_name = self.repo + '.tests_driver_eval.mono'
        self.repo = self.repo + '.sellar'
        if with_coupling:
            proc_name = 'test_mono_driver_sample_generator_sellar_coupling'
        else:
            proc_name = 'test_mono_driver_sample_generator_sellar_list'

        usecase_name = 'usecase1_doe_mono'

        # Associated nested subprocess
        if with_coupling:
            sub_process_name = 'test_sellar_coupling'
        else:
            sub_process_name = 'test_sellar_list'

        sub_process_usecase_name = 'usecase'

        # Creation of the study from the associated usecase
        self.study_name = usecase_name
        imported_module = import_module(
            '.'.join([repo_name, proc_name, usecase_name]))

        study_dump = imported_module.Study(run_usecase=True)

        study_dump.load_data()

        # study_dump.run() # I remove run to be as in GUI test

        # Check the created study

        self.exec_eng = study_dump.ee

        self.exec_eng.display_treeview_nodes()

        if with_coupling:
            ref_disc = self.exec_eng.dm.get_disciplines_with_name(
                f'{self.study_name}.Eval.SellarCoupling.Sellar_1')[0]
        else:
            ref_disc = self.exec_eng.dm.get_disciplines_with_name(
                f'{self.study_name}.Eval.subprocess.Sellar_1')[0]

        # In the study creation it is provided x = array([2.])
        target_x = array([2.])
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # Load the anonymized dict from associated selected sub_process
        # Find anonymised dict
        based_on_uc_name = True
        if based_on_uc_name:  # full anonymized dict with numerical keys
            anonymize_input_dict_from_usecase = study_dump.static_load_raw_usecase_data(
                self.repo, sub_process_name, sub_process_usecase_name)
        else:
            if with_coupling:
                # Below was as it was done in the console first
                # ==============================================================
                # anonymize_input_dict_from_usecase = {}
                # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.x'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_1'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_2'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.z'] = array([1., 1.])
                # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.Sellar_Problem.local_dv'] = 10.
                # ==============================================================

                # Here is as it has been modified to be as in the GUI and/or
                # csv anonymised dict import.
                anonymize_input_dict_from_usecase = {}
                anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.x'] = [
                    1.]
                anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_1'] = [
                    1.]
                anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_2'] = [
                    1.]
                anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.z'] = [
                    1., 1.]
                anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.Sellar_Problem.local_dv'] = 10.

            else:
                # Below was as it was done in the console first
                # ==============================================================
                # anonymize_input_dict_from_usecase = {}
                # anonymize_input_dict_from_usecase['<study_ph>.x'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.y_1'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.y_2'] = array([1.])
                # anonymize_input_dict_from_usecase['<study_ph>.z'] = array([1., 1.])
                # anonymize_input_dict_from_usecase['<study_ph>.subprocess.Sellar_Problem.local_dv'] = 10.
                # ==============================================================

                # Here is as it has been modified to be as in the GUI and/or
                # csv anonymised dict import.
                anonymize_input_dict_from_usecase = {}
                anonymize_input_dict_from_usecase['<study_ph>.subprocess.x'] = [
                    1.]
                anonymize_input_dict_from_usecase['<study_ph>.subprocess.y_1'] = [
                    1.]
                anonymize_input_dict_from_usecase['<study_ph>.subprocess.y_2'] = [
                    1.]
                anonymize_input_dict_from_usecase['<study_ph>.subprocess.z'] = [
                    1., 1.]
                anonymize_input_dict_from_usecase['<study_ph>.subprocess.Sellar_Problem.local_dv'] = 10.

        # Update the reference from the selected imported usecase anonymised
        # dict
        dict_values = {}
        if self.with_modal:
            process_builder_parameter_type = ProcessBuilderParameterType(
                self.repo, sub_process_name, sub_process_usecase_name)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            # process_builder_parameter_type.usecase_data = {}
            dict_values[
                f'{self.study_name}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            dict_values[f'{self.study_name}.Eval.usecase_data'] = anonymize_input_dict_from_usecase

        study_dump.load_data(from_input_dict=dict_values)

        # Check that the reference has been updated

        # In the anonymised dict of the selected usecase it is provided x =
        # array([1.])
        target_x = array([1.])
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # study_dump.run()

    def test_7_usecase_import_multi_instances_eval_generator_cp_sellar_flatten(self):
        """
        This test checks the usecase import capability in multi instance mode.
        """
        from os.path import join, dirname
        ref_dir = join(dirname(__file__), 'data')
        join(ref_dir, 'dump_load_cache')

        repo_name = self.repo + '.tests_driver_eval.multi'
        self.repo = self.repo + '.sellar'
        proc_name = 'test_multi_driver_sample_generator_sellar_coupling'
        usecase_name = 'usecase1_cp_multi_with_ref'

        # Associated nested subprocess

        sub_process_name = 'test_sellar_coupling'
        sub_process_usecase_name = 'usecase'

        # Creation of the study from the associated usecase
        self.study_name = usecase_name
        imported_module = import_module(
            '.'.join([repo_name, proc_name, usecase_name]))

        study_dump = imported_module.Study(run_usecase=True)

        study_dump.load_data()

        study_dump.run()

        # Check the created study

        self.exec_eng = study_dump.ee

        self.exec_eng.display_treeview_nodes()

        ref_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.Eval.ReferenceScenario.SellarCoupling.Sellar_1')[0]

        # In the study creation it is provided x = array([2.])
        target_x = array([2.])
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # Load the anonymized dict from associated selected sub_process

        based_on_uc_name = True
        if based_on_uc_name:  # full anonymized dict with numerical keys
            anonymize_input_dict_from_usecase = study_dump.static_load_raw_usecase_data(
                self.repo, sub_process_name, sub_process_usecase_name)
        else:
            # Below was as it was done in the console first
            # ==================================================================
            # anonymize_input_dict_from_usecase = {}
            # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.x'] = array([1.])
            # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_1'] = array([1.])
            # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_2'] = array([1.])
            # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.z'] = array([1., 1.])
            # anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.Sellar_Problem.local_dv'] = 10.
            # ==================================================================

            # Here is as it has been modified to be as in the GUI and/or
            # csv anonymised dict import
            anonymize_input_dict_from_usecase = {}
            anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.x'] = [
                1.]
            anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_1'] = [
                1.]
            anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_2'] = [
                1.]
            anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.z'] = [
                1., 1.]
            anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.Sellar_Problem.local_dv'] = 10.

        # Update the reference from the selected imported usecase anonymised
        # dict

        dict_values = {}
        if self.with_modal:
            process_builder_parameter_type = ProcessBuilderParameterType(
                self.repo, sub_process_name, sub_process_usecase_name)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            # process_builder_parameter_type.usecase_data = {}
            dict_values[
                f'{self.study_name}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            dict_values[f'{self.study_name}.Eval.usecase_data'] = anonymize_input_dict_from_usecase
        study_dump.load_data(from_input_dict=dict_values)

        self.exec_eng.display_treeview_nodes(True)
        # Check that the reference has been updated

        # In the anonymised dict of the selected usecase it is provided x =
        # array([1.])
        target_x = array([1.])
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

    def test_8_usecase_import_multi_instances_basic_Disc1Disc3(self):
        """
        This test checks the usecase import capability in multi instance mode with eval and without generator (very simple MultiScenario)
        It uses the test_disc1_disc3_list nested process 
        """
        # Old test that could be depreciated: already covered by test 1
        from os.path import join, dirname
        ref_dir = join(dirname(__file__), 'data')
        join(ref_dir, 'dump_load_cache')

        # The generator eval process
        repo_name = self.repo + '.tests_driver_eval.multi'
        proc_name = 'test_multi_driver_simple'
        usecase_name = 'usecase_with_ref'

        # Creation of the study from the associated usecase
        self.study_name = usecase_name
        imported_module = import_module(
            '.'.join([repo_name, proc_name, usecase_name]))

        study_dump = imported_module.Study(run_usecase=True)

        study_dump.load_data()

        # study_dump.run() # In GUI it depends if we do run or not

        # Check the created study

        self.exec_eng = study_dump.ee

        self.exec_eng.display_treeview_nodes()
        ref_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.multi_scenarios.ReferenceScenario.Disc1')[0]

        # In the study creation it is provided x = 2.0
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # Load the anonymized dict from associated selected sub_process

        based_on_uc_name = True
        if based_on_uc_name:
            sub_process_repo = self.repo + '.disc1_disc3'
            sub_process_name = 'test_disc1_disc3_list'
            sub_process_usecase_name = 'usecase'
            #  test_multi_instance_basic is not based on its nested sub_proc
            anonymize_input_dict_from_usecase = study_dump.static_load_raw_usecase_data(
                sub_process_repo, sub_process_name, sub_process_usecase_name)
        else:
            anonymize_input_dict_from_usecase = {}
            anonymize_input_dict_from_usecase['<study_ph>.a'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.x'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.z'] = 1.2
            # anonymize_input_dict_from_usecase['<study_ph>.Disc3.z'] = 1.2
            # if we put a wrong key of variable as in the above example then we
            # have an infinite loop
            anonymize_input_dict_from_usecase['<study_ph>.Disc1.b'] = 4.0
            anonymize_input_dict_from_usecase['<study_ph>.Disc3.constant'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.Disc3.power'] = 2

        # Update the reference from the selected imported usecase anonymised
        # dict

        dict_values = {}
        if self.with_modal:
            process_builder_parameter_type = ProcessBuilderParameterType(
                self.repo, sub_process_name, sub_process_usecase_name)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            # process_builder_parameter_type.usecase_data = {}
            dict_values[
                f'{self.study_name}.multi_scenarios.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            dict_values[f'{self.study_name}.multi_scenarios.usecase_data'] = anonymize_input_dict_from_usecase
        study_dump.load_data(from_input_dict=dict_values)

        # Check that the reference has been updated

        # In the anonymised dict of the selected usecase it is provided x = 3.0
        target_x = 3.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

    def test_9_usecase_import_multi_instances_cp_Disc1Disc3(self):
        """
        This test checks the usecase import capability in multi instance mode with cartesian product generator + eval
        It uses the test_disc1_disc3_list nested process 
        """
        # Old test that could be depreciated: already covered by test 2
        from os.path import join, dirname
        ref_dir = join(dirname(__file__), 'data')
        join(ref_dir, 'dump_load_cache')

        # The generator eval process
        repo_name = self.repo + '.tests_driver_eval.multi'
        proc_name = 'test_multi_driver_sample_generator_simple'
        usecase_name = 'usecase_with_ref'

        # Creation of the study from the associated usecase
        self.study_name = usecase_name
        imported_module = import_module(
            '.'.join([repo_name, proc_name, usecase_name]))

        study_dump = imported_module.Study(run_usecase=True)

        study_dump.load_data()

        # study_dump.run() # In GUI it depends if we do run or not

        # Check the created study

        self.exec_eng = study_dump.ee

        self.exec_eng.display_treeview_nodes()
        ref_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.multi_scenarios.ReferenceScenario.Disc1')[0]

        # In the study creation it is provided x = 2.0
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)

        # Load the anonymized dict from associated selected sub_process

        based_on_uc_name = True
        if based_on_uc_name:
            sub_process_repo = self.repo + '.disc1_disc3'
            sub_process_name = 'test_disc1_disc3_list'
            sub_process_usecase_name = 'usecase'
            #  test_multi_instance_basic is not based on its nested sub_proc
            anonymize_input_dict_from_usecase = study_dump.static_load_raw_usecase_data(
                sub_process_repo, sub_process_name, sub_process_usecase_name)
        else:
            anonymize_input_dict_from_usecase = {}
            anonymize_input_dict_from_usecase['<study_ph>.a'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.x'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.z'] = 1.2
            # anonymize_input_dict_from_usecase['<study_ph>.Disc3.z'] = 1.2
            # if we put a wrong key of variable as in the above example then we
            # have an infinite loop
            anonymize_input_dict_from_usecase['<study_ph>.Disc1.b'] = 4.0
            anonymize_input_dict_from_usecase['<study_ph>.Disc3.constant'] = 3.0
            anonymize_input_dict_from_usecase['<study_ph>.Disc3.power'] = 2

        # Update the reference from the selected imported usecase anonymised
        # dict

        dict_values = {}
        if self.with_modal:
            process_builder_parameter_type = ProcessBuilderParameterType(
                self.repo, sub_process_name, sub_process_usecase_name)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            # process_builder_parameter_type.usecase_data = {}
            dict_values[
                f'{self.study_name}.multi_scenarios.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            dict_values[f'{self.study_name}.multi_scenarios.usecase_data'] = anonymize_input_dict_from_usecase
        study_dump.load_data(from_input_dict=dict_values)

        # Check that the reference has been updated

        # In the anonymised dict of the selected usecase it is provided x = 3.0
        target_x = 3.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc, target_values_dict, print_flag=print_flag)


if __name__ == "__main__":
    test = TestSoSimportUsecase()
    test.setUp()
    test.test_5_usecase_import_multi_instances_eval_generator_cp_sellar()
