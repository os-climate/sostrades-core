'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2024/05/16 Copyright 2023 Capgemini

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
from logging import Handler
from os.path import dirname, join

import pandas as pd
from numpy import allclose, array, ndarray
from pandas._testing import assert_frame_equal

from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import (
    SampleGeneratorWrapper,
)
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import (
    DoeSampleGenerator,
)

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for doe scenario
"""


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestSampleGeneratorWrapper(unittest.TestCase):

    def setUp(self):
        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = "test_sample_generator"

        self.setUp_doe()
        self.setUp_cp()

    def setUp_doe(self):

        self.sampling_method_doe = 'doe_algo'
        self.sampling_generation_mode_doe = 'at_run_time'
        self.study_name_doe = 'doe'

        self.ref_dir = join(dirname(__file__), 'data')

        self.sc_name = "SellarDoeScenario"
        self.c_name = "SellarCoupling"

        dspace_dict = {'variable': ['x', 'z_in', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}

        dspace_dict_optim = {'variable': ['x', 'z_in', 'y_1', 'y_2'],
                             'value': [[1.], [5., 2.], [1.], [1.]],
                             'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                             'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                             'enable_variable': [True, True, True, True],
                             'activated_elem': [[True], [True, True], [True], [True]]}

        dspace_dict_eval = {'variable': ['x', 'z_in'],
                            'lower_bnd': [[0.], [-10., 0.]],
                            'upper_bnd': [[10.], [10., 10.]]
                            }

        self.dspace = pd.DataFrame(dspace_dict)
        self.dspace_eval = pd.DataFrame(dspace_dict_eval)
        self.dspace_optim = pd.DataFrame(dspace_dict_optim)

        input_selection_local_dv_x = {'selected_input': [True, True, False, False, False],
                                      'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                    'y_2',
                                                    'z_in']}
        self.input_selection_local_dv_x = pd.DataFrame(
            input_selection_local_dv_x)

        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z_in']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        input_selection_x = {'selected_input': [False, True, False, False, False],
                             'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                           'y_2',
                                           'z_in']}
        self.input_selection_x = pd.DataFrame(input_selection_x)

        input_selection_local_dv = {'selected_input': [True, False, False, False, False],
                                    'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                  'y_2',
                                                  'z_in']}
        self.input_selection_local_dv = pd.DataFrame(input_selection_local_dv)

        output_selection_obj = {'selected_output': [False, False, True, False, False],
                                'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj = pd.DataFrame(output_selection_obj)

        output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
                                      'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj_y1_y2 = pd.DataFrame(
            output_selection_obj_y1_y2)

    def setUp_cp(self):

        self.sampling_generation_mode_cp = 'at_configuration_time'
        # self.sampling_generation_mode_cp = 'at_run_time'

        self.study_name_cp = 'cp'
        self.sampling_method_cp = 'cartesian_product'

        dict_of_list_values = {
            'x': [0., 3., 4., 5., 7.],
            'y_1': [1.0, 2.0],
            'z': [[-10., 0.], [-5., 4.], [10, 10]]
        }
        list_of_values_x_z = [[], dict_of_list_values['x'],
                              [], [], dict_of_list_values['z']]

        input_selection_cp_x_z = {'selected_input': [False, True, False, False, True],
                                  'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                'y_2',
                                                'z'],
                                  'list_of_values': list_of_values_x_z
                                  }
        self.input_selection_cp_x_z = pd.DataFrame(input_selection_cp_x_z)

        list_of_values_x_y_1_z = [[], dict_of_list_values['x'],
                                  dict_of_list_values['y_1'], [], dict_of_list_values['z']]

        input_selection_cp_x_y_1_z = {'selected_input': [False, True, True, False, True],
                                      'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                    'y_2',
                                                    'z'],
                                      'list_of_values': list_of_values_x_y_1_z
                                      }
        self.input_selection_cp_x_y_1_z = pd.DataFrame(
            input_selection_cp_x_y_1_z)

        input_selection_cp_x = {'selected_input': [False, True, False, False, True],
                                'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                              'y_2',
                                              ''],
                                'list_of_values': list_of_values_x_z
                                }
        self.input_selection_cp_x = pd.DataFrame(input_selection_cp_x)

    """
    These tests are oriented to testing sampling generation in "standalone": i.e. only the sample_generator_wrapper.
    Here there is no nested sub_process and so no evaluation of the sample on a sub_process
    1) In case of doe_algo method
    These tests are oriented to testing sampling generation of only DoE sampling generation. Three families of DoE 
    Gemseo algorithms, classified accordingly by their default options, will be tested:
    - CustomDOE and DiagonalDOE: which shall not work and raise an exception.
    - OT algo family: ['OT_SOBOL', 'OT_RANDOM', 'OT_HASELGROVE', 'OT_REVERSE_HALTON', 'OT_HALTON', 'OT_FAURE', 
                       'OT_MONTE_CARLO', 'OT_FACTORIAL', 'OT_COMPOSITE', 'OT_AXIAL', 'OT_OPT_LHS', 'OT_LHS', 
                       'OT_LHSC', 'OT_FULLFACT', 'OT_SOBOL_INDICES']
    - pydoe family: ['fullfact', 'ff2n', 'pbdesign', 'bbdesign', 'ccdesign', 'lhs']
    2) In case of cartesian product method
    
    """

    def test_1_doe_execution_fullfact(self):
        """
        This is a test of in which a fullfact sampling generation is compared with respect to a
        reference sampling for such DoE algo
        Remark: 
        This is a test converted from EEV3 in which a fullfact sampling generation is compared with respect to a
        reference sampling for such DoE algo. It was made through a DoeEval but now just through a DoE.
        """

        self.ns = f'{self.study_name_doe}'
        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        dspace_dict_eval = {'variable': ['x', 'z'],
                            'lower_bnd': [[0.], [-10., 0.]],
                            'upper_bnd': [[10.], [10., 10.]]
                            }
        self.dspace_eval = pd.DataFrame(dspace_dict_eval)

        exec_eng = ExecutionEngine(self.study_name_doe)

        doe_builder = exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            [doe_builder])

        exec_eng.configure()

        # -- set up disciplines in Scenario
        # DoE inputs
        disc_dict = {}
        n_samples = 10
        disc_dict[f'{self.ns}.SampleGenerator.sampling_method'] = self.sampling_method_doe
        disc_dict[f'{self.ns}.SampleGenerator.sampling_generation_mode'] = self.sampling_generation_mode_doe
        disc_dict[f'{self.ns}.SampleGenerator.sampling_algo'] = "fullfact"
        disc_dict[f'{self.ns}.SampleGenerator.design_space'] = self.dspace_eval
        disc_dict[f'{self.ns}.SampleGenerator.algo_options'] = {
            'n_samples': n_samples, 'fake_option': 'fake_option'}
        disc_dict[f'{self.ns}.SampleGenerator.eval_inputs'] = self.input_selection_x_z

        exec_eng.load_study_from_input_dict(disc_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       '\t|_ SampleGenerator']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        # doe_disc = exec_eng.dm.get_disciplines_with_name(
        #     'doe.DoE')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp
        doe_disc = exec_eng.root_process.proxy_disciplines[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_df')

        dimension = sum([len(sublist) if isinstance(
            sublist, list) else 1 for sublist in list(self.dspace_eval['lower_bnd'].values)])

        theoretical_fullfact_levels = int(n_samples ** (1.0 / dimension))

        theoretical_fullfact_samples = theoretical_fullfact_levels ** dimension
        self.assertEqual(len(doe_disc_samples),
                         theoretical_fullfact_samples)

        # print(doe_disc_samples)
        # test output 'samples_df' sample dataframe
        self.eval_inputs = self.input_selection_x_z
        selected_inputs = self.eval_inputs[self.eval_inputs['selected_input']]['full_name']
        selected_inputs = selected_inputs.tolist()
        target_samples = [[array([0.]), array([-10., 0.])],
                          [array([10.]), array([-10., 0.])],
                          [array([0.]), array([10., 0.])],
                          [array([10.]), array([10., 0.])],
                          [array([0.]), array([-10., 10.])],
                          [array([10.]), array([-10., 10.])],
                          [array([0.]), array([10., 10.])],
                          [array([10.]), array([10., 10.])]]

        target_samples_df = pd.DataFrame(data=target_samples,
                                         columns=selected_inputs)
        # keep only variables columns in samples_df
        doe_disc_samples = doe_disc_samples.drop(
            [SampleGeneratorWrapper.SELECTED_SCENARIO, SampleGeneratorWrapper.SCENARIO_NAME], axis='columns')

        assert_frame_equal(doe_disc_samples, target_samples_df)

    def test_2_doe_pydoe_algo_check(self):
        """
        The aim of this test is to examine DoE Gemseo algorithms which are not CustomDOE, DiagonalDOE or
        OT, that is, pydoe DoE algorithm family.
        This way, a CSV file of the sampling will be generated, analysed and validated so that it is used as reference
        sampling to test the different DoE algorithms aimed by this test.
        """
        self.ns = f'{self.study_name_doe}'
        pydoe_list_of_algo_names = ['fullfact', 'ff2n',
                                    'pbdesign', 'bbdesign', 'ccdesign', 'lhs']

        pydoe_algo_used_options = {'alpha': 'orthogonal',
                                   'face': 'faced',
                                   'criterion': None,
                                   'iterations': 5,
                                   'eval_jac': False,
                                   'center_bb': None,
                                   # center cc shall not be None for ccdesign
                                   # sampling.
                                   'center_cc': (2, 2),
                                   # n_samples shall not be None for fullfact
                                   # sampling.
                                   'n_samples': 5,
                                   'levels': None,
                                   'n_processes': 1,
                                   'wait_time_between_samples': 0.0,
                                   'seed': 1,
                                   'max_time': 0}

        # To work, DoE needs (statically) a sampling_algo and an eval_inputs and (dynamically) a design space.
        # The eval_inputs and design space will be defined below and samplings will be checked to assert that the right
        # columns in the sampling are generated and that they are within design
        # space range.
        input_selection_x_z = {'selected_input': [False, False, False, True, False, True],
                               'full_name': ['u', 'v', 'w', 'x', 'y', 'z']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        dspace_dict_eval = {'variable': ['x', 'z'],
                            'lower_bnd': [[0.], [-10., 0.]],
                            'upper_bnd': [[10.], [10., 10.]]
                            }
        self.dspace_eval = pd.DataFrame(dspace_dict_eval)

        # Build of the DoE discipline under the root process node.
        exec_eng = ExecutionEngine(self.study_name_doe)  # doe

        doe_builder = exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            [doe_builder])

        exec_eng.configure()

        # Check of the proper treeview
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       '\t|_ SampleGenerator']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # DoE inputs. The same algo options will be used for all samplings (let it be noted some default options values
        # will not allow to execute a DoE sampling, e.g. n_samples cannot be
        # equal to None).
        disc_dict = {}
        disc_dict[f'{self.ns}.SampleGenerator.sampling_method'] = "doe_algo"
        disc_dict[f'{self.ns}.SampleGenerator.sampling_generation_mode'] = self.sampling_generation_mode_doe
        disc_dict[f'{self.ns}.SampleGenerator.eval_inputs'] = self.input_selection_x_z
        disc_dict[f'{self.ns}.SampleGenerator.design_space'] = self.dspace_eval
        disc_dict[f'{self.ns}.SampleGenerator.algo_options'] = pydoe_algo_used_options

        # for loop over the sampling algo names and execution to save the
        # resultant sampling in a CSV file.
        name_of_csv = "pydoe_reference_dataframe.csv"
        # f = open(name_of_csv, "w")  # For creating the file
        # f = open(name_of_csv, "r")   # For using the file
        # Extraction of dataframe from reference csv file (use if already
        # created)
        reference_dataframe = pd.read_csv(
            join(self.ref_dir, name_of_csv), sep='\t')
        for sampling_algo_name in pydoe_list_of_algo_names:
            disc_dict[f'{self.ns}.SampleGenerator.sampling_algo'] = sampling_algo_name

            # To check what it is indispensable for each algo.
            # if sampling_algo_name == 'fullfact':
            #     disc_dict[f'{self.ns}.Sample_Generator.algo_options']['n_samples'] = 5
            # else:
            #     disc_dict[f'{self.ns}.Sample_Generator.algo_options']['n_samples'] = None
            # if sampling_algo_name == 'ccdesign':
            #     disc_dict[f'{self.ns}.Sample_Generator.algo_options']['center_cc'] = (2, 2)

            exec_eng.load_study_from_input_dict(disc_dict)

            exec_eng.execute()

            doe_disc = exec_eng.root_process.proxy_disciplines[0]
            doe_disc_samples = doe_disc.get_sosdisc_outputs('samples_df')

            # # Creation of CSV file for reference sampling (use if updating reference dataframe):
            # doe_disc_samples['algo'] = sampling_algo_name
            # doe_disc_samples.to_csv(name_of_csv,
            #                         mode='a',
            #                         sep='\t',
            #                         encoding='utf-8',
            #                         index=False
            #                         )

            # Check whether samples correspond to design space variable
            # selection
            design_space = exec_eng.dm.get_value(
                'doe.SampleGenerator.design_space')

            # keep only variables columns in samples_df
            doe_disc_samples = doe_disc_samples.drop(
                [SampleGeneratorWrapper.SELECTED_SCENARIO, SampleGeneratorWrapper.SCENARIO_NAME], axis='columns')

            self.assertEqual(doe_disc_samples.columns.to_list(),
                             design_space['variable'].to_list())
            # Check whether samples correspond with eval_inputs variable
            # selection
            eval_inputs = exec_eng.dm.get_value(
                'doe.SampleGenerator.eval_inputs')
            self.assertEqual(doe_disc_samples.columns.to_list(),
                             eval_inputs.loc[eval_inputs['selected_input']]['full_name'].to_list())
            # Check whether samples correspond to reference samples
            # Fix format dataframe from CSV file
            algo_reference_samples = reference_dataframe.loc[reference_dataframe['algo']
                                                             == sampling_algo_name]
            algo_reference_samples = algo_reference_samples.reset_index()
            reference_samples = algo_reference_samples[doe_disc_samples.columns.to_list(
            )]
            # Actual check samples correspond to reference samples
            for name, ref_value in reference_samples.to_dict('records')[0].items():
                doe_value = doe_disc_samples.to_dict('records')[0][name]
                nb_list = ref_value[1:-1].split()
                ref_value_list = [float(num) for num in nb_list]
                flag = allclose(ref_value_list, list(doe_value))
                if isinstance(flag, (list, ndarray)):
                    flag = flag.all()
                self.assertTrue(flag)

        # f.close()

    def test_3_doe_OT_algo_check(self):
        """
        The aim of this test is to examine DoE Gemseo algorithms which are not CustomDOE, DiagonalDOE or
        pydoe, that is, OT DoE algorithm family.
        This way, a CSV file of the sampling will be generated, analysed and validated so that it is used as reference
        sampling to test the different DoE algorithms aimed by this test.
        """

        self.ns = f'{self.study_name_doe}'
        # TO FIX OT_RANDOM, OT_MONTE_CARLO, are not reproductible
        OT_list_of_algo_names = ['OT_SOBOL', 'OT_HASELGROVE', 'OT_REVERSE_HALTON', 'OT_HALTON',
                                 'OT_FAURE', 'OT_FACTORIAL', 'OT_COMPOSITE', 'OT_AXIAL',
                                 'OT_OPT_LHS', 'OT_LHS', 'OT_LHSC', 'OT_FULLFACT', 'OT_SOBOL_INDICES']

        OT_algo_default_options = {'levels': None,
                                   'centers': None,
                                   'eval_jac': False,
                                   'eval_second_order': False,
                                   'n_samples': None,
                                   'n_processes': 1,
                                   'wait_time_between_samples': 0.0,
                                   'criterion': 'C2',
                                   'temperature': 'Geometric',
                                   'annealing': True,
                                   'n_replicates': 1000,
                                   'seed': 1,
                                   'max_time': 0}

        OT_algo_used_options = {'levels': None,
                                'centers': None,
                                'eval_jac': False,
                                'eval_second_order': False,
                                # Must be non null at least for OT_SOBOL.
                                'n_samples': 10,
                                'n_processes': 1,
                                'wait_time_between_samples': 0.0,
                                'criterion': 'C2',
                                'temperature': 'Geometric',
                                'annealing': True,
                                'n_replicates': 1000,
                                'seed': 1,
                                'max_time': 0}

        sample_generator = DoeSampleGenerator()

        # To work, DoE needs (statically) a sampling_algo and an eval_inputs and (dinamically) a design space.
        # The eval_inputs and design space will be defined below and samplings will be checked to assert that the right
        # columns in the sampling are generated and that they are within design
        # space range.
        input_selection_x_z = {'selected_input': [False, False, False, True, False, True],
                               'full_name': ['u', 'v', 'w', 'x', 'y', 'z']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        dspace_dict_eval = {'variable': ['x', 'z'],
                            'lower_bnd': [[0.], [-10., 0.]],
                            'upper_bnd': [[10.], [10., 10.]]
                            }
        self.dspace_eval = pd.DataFrame(dspace_dict_eval)

        # Build of the DoE discipline under the root process node.
        exec_eng = ExecutionEngine(self.study_name_doe)  # doe

        doe_builder = exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            [doe_builder])

        exec_eng.configure()

        # Check of the proper treeview
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       '\t|_ SampleGenerator']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # DoE inputs. The same algo options will be used for all samplings (let it be noted some default options values
        # will not allow to execute a DoE sampling, e.g. n_samples cannot be
        # equal to None).
        disc_dict = {}
        disc_dict[f'{self.ns}.SampleGenerator.sampling_method'] = "doe_algo"
        disc_dict[f'{self.ns}.SampleGenerator.sampling_generation_mode'] = self.sampling_generation_mode_doe
        disc_dict[f'{self.ns}.SampleGenerator.eval_inputs'] = self.input_selection_x_z
        disc_dict[f'{self.ns}.SampleGenerator.design_space'] = self.dspace_eval
        disc_dict[f'{self.ns}.SampleGenerator.algo_options'] = OT_algo_used_options

        # for loop over the sampling algo names and execution to save the
        # resultant sampling in a CSV file.
        name_of_csv = "ot_reference_dataframe.csv"
        # f = open(name_of_csv, "w")  # For creating the file
        # f = open(name_of_csv, "r")   # For using the file
        # Extraction of dataframe from reference csv file (use if already
        # created)
        reference_dataframe = pd.read_csv(
            join(self.ref_dir, name_of_csv), sep='\t')
        for sampling_algo_name in OT_list_of_algo_names:
            disc_dict[f'{self.ns}.SampleGenerator.sampling_algo'] = sampling_algo_name

            # To check what it is indispensable for each algo.
            if sampling_algo_name in ['OT_FACTORIAL', 'OT_COMPOSITE', 'OT_AXIAL']:
                disc_dict[f'{self.ns}.SampleGenerator.algo_options']['levels'] = [
                    0.1]  # Must be number between 0 and 1
                disc_dict[f'{self.ns}.SampleGenerator.algo_options']['centers'] = (
                    0, 0, 0)
            else:
                disc_dict[f'{self.ns}.SampleGenerator.algo_options']['levels'] = None
                disc_dict[f'{self.ns}.SampleGenerator.algo_options']['centers'] = None

            exec_eng.load_study_from_input_dict(disc_dict)

            exec_eng.execute()

            doe_disc = exec_eng.root_process.proxy_disciplines[0]
            doe_disc_samples = doe_disc.get_sosdisc_outputs('samples_df')

            # # Creation of CSV file for reference sampling (use if updating reference dataframe):
            # doe_disc_samples['algo'] = sampling_algo_name
            # doe_disc_samples.to_csv(name_of_csv,
            #                         mode='a',
            #                         sep='\t',
            #                         encoding='utf-8',
            #                         index=False
            #                         )

            # Check whether samples correspond to design space variable
            # selection
            design_space = exec_eng.dm.get_value(
                'doe.SampleGenerator.design_space')
            # remove scenario columns from samples_df
            doe_disc_samples = doe_disc_samples.drop(
                [SampleGeneratorWrapper.SELECTED_SCENARIO, SampleGeneratorWrapper.SCENARIO_NAME], axis='columns')

            self.assertEqual(doe_disc_samples.columns.to_list(),
                             design_space['variable'].to_list())
            # Check whether samples correspond with eval_inputs variable
            # selection
            eval_inputs = exec_eng.dm.get_value(
                'doe.SampleGenerator.eval_inputs')
            self.assertEqual(doe_disc_samples.columns.to_list(),
                             eval_inputs.loc[eval_inputs['selected_input']]['full_name'].to_list())
            # Check whether samples correspond to reference samples
            # Fix format dataframe from CSV file
            algo_reference_samples = reference_dataframe.loc[reference_dataframe['algo']
                                                             == sampling_algo_name]
            algo_reference_samples = algo_reference_samples.reset_index()
            reference_samples = algo_reference_samples[doe_disc_samples.columns.to_list(
            )]
            # Actual check samples correspond to reference samples
            for name, ref_value in reference_samples.to_dict('records')[0].items():
                doe_value = doe_disc_samples.to_dict('records')[0][name]
                nb_list = ref_value[1:-1].split()
                ref_value_list = [float(num) for num in nb_list]
                flag = allclose(ref_value_list, list(doe_value))
                if isinstance(flag, (list, ndarray)):
                    flag = flag.all()
                self.assertTrue(flag, f'The algo {sampling_algo_name} does not give same value as reference file')
        # f.close()

    def _test_4_cartesian_product_execution(self):
        # NB: test no longer stands as standalone sample generator sampling at config. time is deactivated
        """
        This is a test of the cartesian product wrapper
        """
        self.ns = f'{self.study_name_cp}'
        exec_eng = ExecutionEngine(self.study_name_cp)

        doe_builder = exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            [doe_builder])

        exec_eng.configure()

        #

        # -- set up disciplines in Scenario
        # CP inputs
        disc_dict = {}
        disc_dict[f'{self.ns}.SampleGenerator.sampling_method'] = 'cartesian_product'
        disc_dict[f'{self.ns}.SampleGenerator.eval_inputs'] = self.input_selection_cp_x_z
        # disc_dict[f'{self.ns}.CP.generated_samples'] = generated_samples

        exec_eng.load_study_from_input_dict(disc_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ cp',
                       '\t|_ SampleGenerator']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        exec_eng.display_treeview_nodes(True)

        disc = exec_eng.root_process.proxy_disciplines[0]
        disc_sampling_method = disc.get_sosdisc_inputs(
            'sampling_method')
        print('sampling__method:')
        print(disc_sampling_method)

        disc_eval_inputs_cp = disc.get_sosdisc_inputs(
            'eval_inputs')
        print('eval_inputs_cp 2:')
        print(disc_eval_inputs_cp)

        # if self.sampling_generation_mode_cp == 'at_configuration_time':
        #     disc_generated_samples = disc.get_sosdisc_inputs(
        #         'generated_samples')
        #     print('generated_samples:')
        #     print(disc_generated_samples)

        exec_eng.execute()

        # disc = exec_eng.dm.get_disciplines_with_name(
        #     'cp.CP')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp
        disc = exec_eng.root_process.proxy_disciplines[0]

        if self.sampling_generation_mode_cp == 'at_run_time':
            disc_samples = disc.get_sosdisc_outputs(
                'samples_df')

            print(disc_samples)

        targeted_samples = [
            [0.0, [-10.0, 0.0]],
            [0.0, [-5.0, 4.0]],
            [0.0, [10, 10]],
            [3.0, [-10.0, 0.0]],
            [3.0, [-5.0, 4.0]],
            [3.0, [10, 10]],
            [4.0, [-10.0, 0.0]],
            [4.0, [-5.0, 4.0]],
            [4.0, [10, 10]],
            [5.0, [-10.0, 0.0]],
            [5.0, [-5.0, 4.0]],
            [5.0, [10, 10]],
            [7.0, [-10.0, 0.0]],
            [7.0, [-5.0, 4.0]],
            [7.0, [10, 10]]]

        variable_list = ['x', 'z']
        target_samples_df = pd.DataFrame(targeted_samples, columns=variable_list)

    def _test_5_cartesian_product_step_by_step_execution(self):
        """
        This is a test of the cartesian product wrapper
        """
        # NB: test no longer stands as standalone sample generator sampling at config. time is to  deactivated
        self.ns = f'{self.study_name_cp}'
        exec_eng = ExecutionEngine(self.study_name_cp)

        doe_builder = exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            [doe_builder])

        exec_eng.configure()

        #
        # -- set up disciplines in Scenario
        # 1. Input sampling_method
        # CP inputs
        disc_dict = {}
        disc_dict[f'{self.ns}.SampleGenerator.sampling_method'] = 'cartesian_product'

        exec_eng.load_study_from_input_dict(disc_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ cp',
                       '\t|_ SampleGenerator']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        exec_eng.display_treeview_nodes(True)

        # disc = exec_eng.dm.get_disciplines_with_name(
        #     'cp.CP')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp
        disc = exec_eng.root_process.proxy_disciplines[0]

        disc_sampling_method = disc.get_sosdisc_inputs(
            'sampling_method')
        print('sampling__method:')
        print(disc_sampling_method)

        disc_eval_inputs_cp = disc.get_sosdisc_inputs(
            'eval_inputs')
        print('eval_inputs_cp 1:')
        print(disc_eval_inputs_cp)

        # 2. Input eval_inputs_cp
        # CP inputs
        disc_dict = {}
        disc_dict[f'{self.ns}.SampleGenerator.eval_inputs'] = self.input_selection_cp_x_z
        exec_eng.load_study_from_input_dict(disc_dict)

        exec_eng.display_treeview_nodes(True)

        disc_sampling_method = disc.get_sosdisc_inputs(
            'sampling_method')
        print('sampling__method:')
        print(disc_sampling_method)

        disc_eval_inputs_cp = disc.get_sosdisc_inputs(
            'eval_inputs')
        print('eval_inputs_cp 2:')
        print(disc_eval_inputs_cp)

        # if self.sampling_generation_mode_cp == 'at_configuration_time':
        #     disc_generated_samples = disc.get_sosdisc_inputs(
        #         'generated_samples')
        #     print('generated_samples:')
        #     print(disc_generated_samples)

        # 3. Input an updated eval_inputs_cp
        # CP inputs
        disc_dict = {}
        disc_dict[f'{self.ns}.SampleGenerator.eval_inputs'] = self.input_selection_cp_x_y_1_z
        exec_eng.load_study_from_input_dict(disc_dict)

        exec_eng.display_treeview_nodes(True)

        disc_sampling_method = disc.get_sosdisc_inputs(
            'sampling_method')
        print('sampling__method:')
        print(disc_sampling_method)

        disc_eval_inputs_cp = disc.get_sosdisc_inputs(
            'eval_inputs')
        print('eval_inputs_cp 3:')
        print(disc_eval_inputs_cp)

        # if self.sampling_generation_mode_cp == 'at_configuration_time':
        #     disc_generated_samples = disc.get_sosdisc_inputs(
        #         'generated_samples')
        #     print('generated_samples:')
        #     print(disc_generated_samples)

        exec_eng.execute()

        if self.sampling_generation_mode_cp == 'at_run_time':
            disc_samples = disc.get_sosdisc_outputs(
                'samples_df')

            print(disc_samples)

        # 4. Change sampling_method: go to doe_algo
        # CP inputs
        disc_dict = {}
        disc_dict[f'{self.ns}.SampleGenerator.sampling_method'] = 'doe_algo'

        exec_eng.load_study_from_input_dict(disc_dict)

        # 5. Change sampling_method: come back  to cartesian_product
        # CP inputs
        disc_dict = {}
        disc_dict[f'{self.ns}.SampleGenerator.sampling_method'] = 'cartesian_product'

        exec_eng.load_study_from_input_dict(disc_dict)

        disc_dict = {}
        disc_dict[f'{self.ns}.SampleGenerator.eval_inputs'] = self.input_selection_cp_x_z
        exec_eng.load_study_from_input_dict(disc_dict)

        disc_dict = {}
        disc_dict[f'{self.ns}.SampleGenerator.eval_inputs'] = self.input_selection_cp_x_y_1_z
        exec_eng.load_study_from_input_dict(disc_dict)

        exec_eng.execute()
