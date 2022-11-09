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
import logging
from logging import Handler
from time import time

from pandas._testing import assert_frame_equal

from gemseo.algos.doe.doe_factory import DOEFactory

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for doe scenario
"""

import unittest
from numpy import array
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import DoeSampleGenerator
import os
from os.path import dirname, join


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestSoSDOEScenario(unittest.TestCase):

    def setUp(self):
        self.study_name = 'doe'
        self.ns = f'{self.study_name}'
        self.sc_name = "SellarDoeScenario"
        self.c_name = "SellarCoupling"
        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}

        dspace_dict_optim = {'variable': ['x', 'z', 'y_1', 'y_2'],
                             'value': [[1.], [5., 2.], [1.], [1.]],
                             'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                             'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                             'enable_variable': [True, True, True, True],
                             'activated_elem': [[True], [True, True], [True], [True]]}

        dspace_dict_eval = {'variable': ['x', 'z'],
                            'lower_bnd': [[0.], [-10., 0.]],
                            'upper_bnd': [[10.], [10., 10.]]
                            }

        self.dspace = pd.DataFrame(dspace_dict)
        self.dspace_eval = pd.DataFrame(dspace_dict_eval)
        self.dspace_optim = pd.DataFrame(dspace_dict_optim)

        input_selection_local_dv_x = {'selected_input': [True, True, False, False, False],
                                      'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                    'y_2',
                                                    'z']}
        self.input_selection_local_dv_x = pd.DataFrame(
            input_selection_local_dv_x)

        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        input_selection_x = {'selected_input': [False, True, False, False, False],
                             'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                           'y_2',
                                           'z']}
        self.input_selection_x = pd.DataFrame(input_selection_x)

        input_selection_local_dv = {'selected_input': [True, False, False, False, False],
                                    'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                  'y_2',
                                                  'z']}
        self.input_selection_local_dv = pd.DataFrame(input_selection_local_dv)

        output_selection_obj = {'selected_output': [False, False, True, False, False],
                                'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj = pd.DataFrame(output_selection_obj)

        output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
                                      'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj_y1_y2 = pd.DataFrame(
            output_selection_obj_y1_y2)

        self.repo = 'sostrades_core.sos_processes.test'

    def _test_1_doe_eval_execution_fullfact(self):
        """ We test that the number of samples generated by the fullfact algorithm is the theoretical expected number
        Pay attention to the fact that an additional sample (the reference one ) is added
        """

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 10
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "fullfact"
        disc_dict[f'{self.ns}.DoEEval.design_space'] = self.dspace_eval
        disc_dict[f'{self.ns}.DoEEval.algo_options'] = {
            'n_samples': n_samples, 'fake_option': 'fake_option'}
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x_z
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.subprocess.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name(
            'doe.DoEEval')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')

        dimension = sum([len(sublist) if isinstance(
            sublist, list) else 1 for sublist in list(self.dspace_eval['lower_bnd'].values)])

        theoretical_fullfact_levels = int(n_samples ** (1.0 / dimension))

        theoretical_fullfact_samples = theoretical_fullfact_levels ** dimension
        self.assertEqual(len(doe_disc_samples),
                         theoretical_fullfact_samples + 1)

    def _test_2_doe_eval_CustomDoE(self):
        """ this test checks that the doe_eval delivers expected outputs
        It is a non regression test
        """

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {f'{self.ns}.DoEEval.sampling_algo': "CustomDOE",
                     f'{self.ns}.DoEEval.eval_inputs': self.input_selection_x_z,
                     f'{self.ns}.DoEEval.eval_outputs': self.output_selection_obj_y1_y2}
        # DoE inputs

        x_values = [array([9.379763880395856]), array([8.88644794300546]),
                    array([3.7137135749628882]), array([0.0417022004702574]), array([6.954954792150857])]
        z_values = [array([1.515949043849158, 5.6317362409322165]),
                    array([-1.1962705421254114, 6.523436208612142]),
                    array([-1.9947578026244557, 4.822570933860785]
                          ), array([1.7490668861813, 3.617234050834533]),
                    array([-9.316161097119341, 9.918161285133076])]

        wrong_values = 5 * [0.0]

        samples_dict = {'x': x_values, 'z': z_values,'wrong_values':wrong_values}
        # samples_dict = {'z': z_values, 'x': x_values,
        #                 'wrong_values': wrong_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{self.ns}.DoEEval.samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {f'{self.ns}.x': array([1.]), f'{self.ns}.y_1': array([1.]), f'{self.ns}.y_2': array([1.]),
                       f'{self.ns}.z': array([1., 1.]), f'{self.ns}.DoEEval.subprocess.Sellar_Problem.local_dv': local_dv}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name(
            'doe.DoEEval')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        doe_disc_y1 = doe_disc.get_sosdisc_outputs('y_1_dict')
        doe_disc_y2 = doe_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(doe_disc_samples), 6)
        self.assertEqual(len(doe_disc_obj), 6)
        reference_dict_doe_disc_y1 = {'scenario_1': array([15.102817691025274]), 'scenario_2': array([15.000894464408367]),
                                      'scenario_3': array([11.278122259980103]), 'scenario_4': array([5.1893098993071565]),
                                      'scenario_5': array([101.52834810032466]), 'reference': array([2.29689011157193])}
        reference_dict_doe_disc_y2 = {'scenario_1': array([11.033919669249876]), 'scenario_2': array([9.200264485831308]),
                                      'scenario_3': array([6.186104699873589]), 'scenario_4': array([7.644306621667905]),
                                      'scenario_5': array([10.67812782219566]), 'reference': array([3.515549442140351])}
        for key in doe_disc_y1.keys():
            self.assertAlmostEqual(
                doe_disc_y1[key][0], reference_dict_doe_disc_y1[key][0])
        for key in doe_disc_y2.keys():
            self.assertAlmostEqual(
                doe_disc_y2[key][0], reference_dict_doe_disc_y2[key][0])

        # self.assertDictEqual(doe_disc_y1,
        #                      {'scenario_1': array([15.102817691025274]), 'scenario_2': array([15.000894464408367]),
        #                       'scenario_3': array([11.278122259980103]), 'scenario_4': array([5.1893098993071565]),
        #                       'scenario_5': array([101.52834810032466]), 'reference': array([2.29689011157193])})
        # self.assertDictEqual(doe_disc_y2,
        #                      {'scenario_1': array([11.033919669249876]), 'scenario_2': array([9.200264485831308]),
        #                       'scenario_3': array([6.186104699873589]), 'scenario_4': array([7.644306621667905]),
        #                       'scenario_5': array([10.67812782219566]), 'reference': array([3.515549442140351])})

    def test_2_2_Eval_User_Defined_samples_alpha(self):
        """
        This test checks that the custom samples applied to an Eval driver delivers expected outputs
        It is a non regression test
        """
        study_name = 'root'
        ns = study_name

        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_eval"
        eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                        mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            eval_builder)

        exec_eng.configure()

        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['Eval.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
                                      'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj_y1_y2 = pd.DataFrame(
            output_selection_obj_y1_y2)

        # -- set up disciplines in Scenario
        disc_dict = {f'{ns}.Eval.eval_inputs': self.input_selection_x_z,
                     f'{ns}.Eval.eval_outputs': self.output_selection_obj_y1_y2}
        # Eval inputs

        x_values = [array([9.379763880395856]), array([8.88644794300546]),
                    array([3.7137135749628882]), array([0.0417022004702574]), array([6.954954792150857])]
        z_values = [array([1.515949043849158, 5.6317362409322165]),
                    array([-1.1962705421254114, 6.523436208612142]),
                    array([-1.9947578026244557, 4.822570933860785]
                          ), array([1.7490668861813, 3.617234050834533]),
                    array([-9.316161097119341, 9.918161285133076])]

        wrong_values = 5 * [0.0]

        samples_dict = {'x': x_values, 'z': z_values,'wrong_values':wrong_values}
        # samples_dict = {'z': z_values, 'x': x_values,
        #                 'wrong_values': wrong_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{ns}.Eval.samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {f'{ns}.x': array([1.]), f'{ns}.y_1': array([1.]), f'{ns}.y_2': array([1.]),
                       f'{ns}.z': array([1., 1.]),
                       f'{ns}.Eval.subprocess.Sellar_Problem.local_dv': local_dv}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {ns}',
                       '|_ root',
                       f'\t|_ Eval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        root_outputs = exec_eng.root_process.get_output_data_names()
        self.assertIn('root.obj_dict', root_outputs)
        self.assertIn('root.y_1_dict', root_outputs)
        self.assertIn('root.y_2_dict', root_outputs)

        # doe_disc = exec_eng.dm.get_disciplines_with_name(f'{ns}.Eval')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp
        doe_disc = exec_eng.dm.get_disciplines_with_name(f'{ns}.Eval')[0]


        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        doe_disc_y1 = doe_disc.get_sosdisc_outputs('y_1_dict')
        doe_disc_y2 = doe_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(doe_disc_samples), 6)
        self.assertEqual(len(doe_disc_obj), 6)
        reference_dict_doe_disc_y1 = {'scenario_1': array([15.102817691025274]),
                                      'scenario_2': array([15.000894464408367]),
                                      'scenario_3': array([11.278122259980103]),
                                      'scenario_4': array([5.1893098993071565]),
                                      'scenario_5': array([101.52834810032466]), 'reference': array([2.29689011157193])}
        reference_dict_doe_disc_y2 = {'scenario_1': array([11.033919669249876]),
                                      'scenario_2': array([9.200264485831308]),
                                      'scenario_3': array([6.186104699873589]),
                                      'scenario_4': array([7.644306621667905]),
                                      'scenario_5': array([10.67812782219566]), 'reference': array([3.515549442140351])}
        for key in doe_disc_y1.keys():
            self.assertAlmostEqual(doe_disc_y1[key][0], reference_dict_doe_disc_y1[key][0])
        for key in doe_disc_y2.keys():
            self.assertAlmostEqual(doe_disc_y2[key][0], reference_dict_doe_disc_y2[key][0])

    def _test_3_doe_eval_execution_lhs_on_1_var(self):
        """ this test is a non regression test on doe eval
            we run the doe_eval with lhs as sampling algorithm and
            we check that outputs are as expected
        """

        dspace_dict_x = {'variable': ['x'],

                         'lower_bnd': [0.],
                         'upper_bnd': [10.],

                         }
        dspace_x = pd.DataFrame(dspace_dict_x)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 10
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "lhs"
        disc_dict[f'{self.ns}.DoEEval.design_space'] = dspace_x
        disc_dict[f'{self.ns}.DoEEval.algo_options'] = {'n_samples': n_samples}
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.subprocess.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name(
            'doe.DoEEval')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        doe_disc_y1 = doe_disc.get_sosdisc_outputs('y_1_dict')
        doe_disc_y2 = doe_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(doe_disc_samples), n_samples + 1)
        self.assertEqual(len(doe_disc_obj), n_samples + 1)
        reference_dict_doe_disc_y1 = {'scenario_1': array([10.491019856682016]), 'scenario_2': array([7.247824531594309]),
                                      'scenario_3': array([2.9753409599263483]), 'scenario_4': array([1.7522749587335193]),
                                      'scenario_5': array([9.384097972066053]), 'scenario_6': array([8.36704386923391]),
                                      'scenario_7': array([4.479056921478663]), 'scenario_8': array([5.286891081070988]),
                                      'scenario_9': array([3.240108355137796]), 'scenario_10': array([6.194561090631401]),
                                      'reference': array([2.29689011157193])}
        reference_dict_doe_disc_y2 = {'scenario_1': array([5.238984386606706]), 'scenario_2': array([4.692178398916815]),
                                      'scenario_3': array([3.7249176675790494]), 'scenario_4': array([3.3237352298452736]),
                                      'scenario_5': array([5.063347510823095]), 'scenario_6': array([4.892584289045681]),
                                      'scenario_7': array([4.116378255765888]), 'scenario_8': array([4.2993240487306235]),
                                      'scenario_9': array([3.8000300983977455]), 'scenario_10': array([4.488887520686984]),
                                      'reference': array([3.5155494421403515])}
        for key in doe_disc_y1.keys():
            self.assertAlmostEqual(
                doe_disc_y1[key][0], reference_dict_doe_disc_y1[key][0])
        for key in doe_disc_y2.keys():
            self.assertAlmostEqual(
                doe_disc_y2[key][0], reference_dict_doe_disc_y2[key][0])

        # we check that at the end of the run the dm contains the reference (or
        # initial ) point
        self.assertEqual(exec_eng.dm.get_value('doe.x'), array([1.0]))

    def test_3_2_separated_doe_and_eval_execution_lhs_on_1_var(self):
        """
        """
        input_selection_x = {'selected_input': [False, True, False, False, False],
                             'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                           'y_2',
                                           'z']}
        self.input_selection_x = pd.DataFrame(input_selection_x)

        dspace_dict_x = {'variable': ['x'],

                         'lower_bnd': [0.],
                         'upper_bnd': [10.],

                         }
        dspace_x = pd.DataFrame(dspace_dict_x)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval_separated"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 10
        disc_dict[f'{self.ns}.DoE.sampling_algo'] = "lhs"
        disc_dict[f'{self.ns}.DoE.design_space'] = dspace_x
        disc_dict[f'{self.ns}.DoE.algo_options'] = {'n_samples': n_samples}
        disc_dict[f'{self.ns}.eval_inputs'] = self.input_selection_x

        # Eval inputs
        # disc_dict[f'{self.ns}.eval_inputs'] = disc_dict[f'{self.ns}.eval_inputs']
        disc_dict[f'{self.ns}.eval_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.Eval.subprocess.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoE',
                       '\t|_ Eval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        eval_disc = exec_eng.dm.get_disciplines_with_name(
            'doe.Eval')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp

        eval_disc_samples = eval_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        eval_disc_obj = eval_disc.get_sosdisc_outputs('obj_dict')
        eval_disc_y1 = eval_disc.get_sosdisc_outputs('y_1_dict')
        eval_disc_y2 = eval_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(eval_disc_samples), n_samples + 1)
        self.assertEqual(len(eval_disc_obj), n_samples + 1)
        reference_dict_eval_disc_y1 = {'scenario_1': array([10.491019856682016]), 'scenario_2': array([7.247824531594309]),
                                      'scenario_3': array([2.9753409599263483]), 'scenario_4': array([1.7522749587335193]),
                                      'scenario_5': array([9.384097972066053]), 'scenario_6': array([8.36704386923391]),
                                      'scenario_7': array([4.479056921478663]), 'scenario_8': array([5.286891081070988]),
                                      'scenario_9': array([3.240108355137796]), 'scenario_10': array([6.194561090631401]),
                                      'reference': array([2.29689011157193])}
        reference_dict_eval_disc_y2 = {'scenario_1': array([5.238984386606706]), 'scenario_2': array([4.692178398916815]),
                                      'scenario_3': array([3.7249176675790494]), 'scenario_4': array([3.3237352298452736]),
                                      'scenario_5': array([5.063347510823095]), 'scenario_6': array([4.892584289045681]),
                                      'scenario_7': array([4.116378255765888]), 'scenario_8': array([4.2993240487306235]),
                                      'scenario_9': array([3.8000300983977455]), 'scenario_10': array([4.488887520686984]),
                                      'reference': array([3.5155494421403515])}
        for key in eval_disc_y1.keys():
            self.assertAlmostEqual(
                eval_disc_y1[key][0], reference_dict_eval_disc_y1[key][0])
        for key in eval_disc_y2.keys():
            self.assertAlmostEqual(
                eval_disc_y2[key][0], reference_dict_eval_disc_y2[key][0])

        # we check that at the end of the run the dm contains the reference (or
        # initial ) point
        self.assertEqual(exec_eng.dm.get_value('doe.x'), array([1.0]))

    def test_4_doe_eval_options_and_design_space_after_reconfiguration(self):
        """ different configurations of doe eval are tested here The aim is to assert that doe_eval configuration
        runs as intended Different inputs are modified (algo_name, design_space,...) and we check that the dm
        contains the expected values afterward
        """

        dspace_dict_x = {'variable': ['x'],

                         'lower_bnd': [[0.]],
                         'upper_bnd': [[10.]]
                         }
        dspace_x = pd.DataFrame(dspace_dict_x)

        dspace_dict_x_eval = {'variable': ['x'],

                              'lower_bnd': [[5.]],
                              'upper_bnd': [[11.]]
                              }
        dspace_x_eval = pd.DataFrame(dspace_dict_x_eval)

        dspace_dict_x_local_dv = {'variable': ['x', 'DoEEval.subprocess.Sellar_Problem.local_dv'],

                                  'lower_bnd': [[0.], 0.],
                                  'upper_bnd': [[10.], 10.]
                                  }
        dspace_x_local_dv = pd.DataFrame(dspace_dict_x_local_dv)

        dspace_dict_x_z = {'variable': ['x', 'z'],

                           'lower_bnd': [[0.], [0., 0.]],
                           'upper_bnd': [[10.], [10., 10.]]
                           }
        dspace_x_z = pd.DataFrame(dspace_dict_x_z)

        dspace_dict_eval = {'variable': ['x', 'z'],

                            'lower_bnd': [[0.], [-10., 0.]],
                            'upper_bnd': [[10.], [10., 10.]]
                            }
        dspace_eval = pd.DataFrame(dspace_dict_eval)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # -- set up disciplines
        values_dict = {}
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.subprocess.Sellar_Problem.local_dv'] = 10
        exec_eng.load_study_from_input_dict(values_dict)

        # configure disciplines with the algo lhs
        algo_name = "lhs"
        disc_dict = {}
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = algo_name
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj
        exec_eng.load_study_from_input_dict(disc_dict)

        default_algo_options_lhs, algo_options_descr_dict = DoeSampleGenerator().get_options_desc_in(
            algo_name)

        self.assertDictEqual(exec_eng.dm.get_value(
            'doe.DoEEval.algo_options'), default_algo_options_lhs)
        # WARNING: default design space with array is built with 2-elements arrays : [0., 0.]
        # but dspace_x contains 1-element arrays
        #         assert_frame_equal(exec_eng.dm.get_value('doe.DoEEval.design_space').reset_index(drop=True),
        # dspace_x.reset_index(drop=True), check_dtype=False)

        # trigger a reconfiguration after options and design space changes
        n_samples = 10
        face = 'faced'
        disc_dict = {'doe.DoEEval.algo_options': {'n_samples': n_samples, 'face': face},
                     'doe.DoEEval.design_space': dspace_x_eval}
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertEqual(exec_eng.dm.get_value(
            'doe.DoEEval.algo_options')['n_samples'], n_samples)
        self.assertEqual(exec_eng.dm.get_value(
            'doe.DoEEval.algo_options')['face'], face)

        assert_frame_equal(exec_eng.dm.get_value('doe.DoEEval.design_space').reset_index(drop=True),
                           dspace_x_eval.reset_index(drop=True), check_dtype=False)

        # trigger a reconfiguration after algo name change
        algo_name = "fullfact"
        disc_dict = {'doe.DoEEval.sampling_algo': algo_name}
        exec_eng.load_study_from_input_dict(disc_dict)
        default_algo_options_fullfact, algo_options_descr_dict = DoeSampleGenerator().get_options_desc_in(
            algo_name)
        self.assertDictEqual(exec_eng.dm.get_value(
            'doe.DoEEval.algo_options'), default_algo_options_fullfact)
        assert_frame_equal(exec_eng.dm.get_value('doe.DoEEval.design_space').reset_index(drop=True),
                           dspace_x_eval.reset_index(drop=True), check_dtype=False)

        disc_dict = {'doe.DoEEval.algo_options': {
            'n_samples': n_samples, 'face': face}}
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertEqual(exec_eng.dm.get_value(
            'doe.DoEEval.algo_options')['n_samples'], n_samples)
        self.assertEqual(exec_eng.dm.get_value(
            'doe.DoEEval.algo_options')['face'], face)

        # trigger a reconfiguration after eval_inputs and eval_outputs changes
        disc_dict = {'doe.DoEEval.eval_outputs': self.output_selection_obj_y1_y2,
                     'doe.DoEEval.eval_inputs': self.input_selection_x_z}
        exec_eng.load_study_from_input_dict(disc_dict)
        # self.assertDictEqual(exec_eng.dm.get_value(
        #     'doe.DoEEval.algo_options'), default_algo_options_fullfact)
        assert exec_eng.dm.get_value('doe.DoEEval.eval_outputs').equals(
            self.output_selection_obj_y1_y2)
        df1 = self.input_selection_x_z
        df2 = exec_eng.dm.get_value('doe.DoEEval.eval_inputs')
        df_all = df1.merge(df2, on=['selected_input', 'full_name'],
                           how='left', indicator=True)
        assert df_all[['selected_input', 'full_name']].equals(
            self.input_selection_x_z)
        self.assertTrue(exec_eng.dm.get_value('doe.DoEEval.design_space').reset_index(drop=True)['variable'].equals(
            dspace_x_z.reset_index(drop=True)['variable']))

        disc_dict = {'doe.DoEEval.algo_options': {'n_samples': n_samples, 'face': face},
                     'doe.DoEEval.eval_outputs': self.output_selection_obj_y1_y2,
                     'doe.DoEEval.design_space': dspace_eval}
        exec_eng.load_study_from_input_dict(disc_dict)
        algo_full_options = default_algo_options_fullfact
        algo_full_options.update({'n_samples': n_samples, 'face': face})

        self.assertDictEqual(exec_eng.dm.get_value('doe.DoEEval.algo_options'),
                             algo_full_options)
        assert_frame_equal(exec_eng.dm.get_value('doe.DoEEval.design_space').reset_index(drop=True),
                           dspace_eval.reset_index(drop=True), check_dtype=False)
        # exec_eng.execute()

    def _test_5_doe_eval_CustomDoE_reconfiguration(self):
        """ different configurations of custom doe are tested here
        The aim is to assert that doe_eval configuration runs as intended
        Different inputs are modified and we check that the dm contains the expected values afterward
        At the end of the test we check that the generated samples are the ones expected and that
        the dm contains initial values after doe_eval run
        """

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # -- set up disciplines in Scenario
        disc_dict = {f'{self.ns}.DoEEval.sampling_algo': "CustomDOE",
                     f'{self.ns}.DoEEval.eval_inputs': self.input_selection_x,
                     f'{self.ns}.DoEEval.eval_outputs': self.output_selection_obj}
        # DoE inputs
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertListEqual(exec_eng.dm.get_value(
            'doe.DoEEval.samples_df').columns.tolist(), ['x'])
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_local_dv_x
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertListEqual(exec_eng.dm.get_value('doe.DoEEval.samples_df').columns.tolist(),
                             ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x'])
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_local_dv
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertListEqual(exec_eng.dm.get_value('doe.DoEEval.samples_df').columns.tolist(),
                             ['DoEEval.subprocess.Sellar_Problem.local_dv'])
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x_z
        exec_eng.load_study_from_input_dict(disc_dict)

        x_values = [array([9.379763880395856]), array([8.88644794300546]),
                    array([3.7137135749628882]), array([0.0417022004702574]), array([6.954954792150857])]
        z_values = [array([1.515949043849158, 5.6317362409322165]),
                    array([-1.1962705421254114, 6.523436208612142]),
                    array([-1.9947578026244557, 4.822570933860785]
                          ), array([1.7490668861813, 3.617234050834533]),
                    array([-9.316161097119341, 9.918161285133076])]

        samples_dict = {'x': x_values, 'z': z_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{self.ns}.DoEEval.samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {f'{self.ns}.x': array([1.]), f'{self.ns}.y_1': array([1.]), f'{self.ns}.y_2': array([1.]),
                       f'{self.ns}.z': array([1., 1.]), f'{self.ns}.DoEEval.subprocess.Sellar_Problem.local_dv': local_dv}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')

        # check that the generated samples are the ones expected (custom sample
        # + reference value)
        expected_doe_disc_samples = pd.DataFrame(
            {'scenario': ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4', 'scenario_5', 'reference'],
             'x': x_values + [1.000000], 'z': z_values + [array([1.0, 1.0])]})
        assert_frame_equal(
            doe_disc_samples, expected_doe_disc_samples, check_dtype=False)

        # check that at the end of doe eval dm still contains initial
        # (reference) point
        self.assertEqual(exec_eng.dm.get_value('doe.x'), 1.0)
        self.assertEqual(exec_eng.dm.get_value(
            'doe.z').tolist(), array([1., 1.]).tolist())

    def test_5_2_Eval_User_defined_samples_reconfiguration(self):
        """ different configurations of user-defined samples are tested here
        The aim is to assert that eval configuration runs as intended
        Different inputs are modified and we check that the dm contains the expected values afterward
        At the end of the test we check that the generated samples are the ones expected and that
        the dm contains initial values after doe_eval run
        """

        input_selection_x = {'selected_input': [False, True, False, False, False],
                             'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                           'y_2',
                                           'z']}
        self.input_selection_x = pd.DataFrame(input_selection_x)

        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        input_selection_local_dv_x = {'selected_input': [True, True, False, False, False],
                                      'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                    'y_2',
                                                    'z']}
        self.input_selection_local_dv_x = pd.DataFrame(
            input_selection_local_dv_x)

        input_selection_local_dv = {'selected_input': [True, False, False, False, False],
                                    'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                  'y_2',
                                                  'z']}
        self.input_selection_local_dv = pd.DataFrame(input_selection_local_dv)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_eval"
        eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ Eval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # -- set up disciplines in Scenario
        disc_dict = {f'{self.ns}.Eval.eval_inputs': self.input_selection_x,
                     f'{self.ns}.Eval.eval_outputs': self.output_selection_obj}
        # DoE inputs
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertListEqual(exec_eng.dm.get_value(
            'doe.Eval.samples_df').columns.tolist(), ['x'])
        disc_dict[f'{self.ns}.Eval.eval_inputs'] = self.input_selection_local_dv_x
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertListEqual(exec_eng.dm.get_value('doe.Eval.samples_df').columns.tolist(),
                             ['Eval.subprocess.Sellar_Problem.local_dv', 'x'])
        disc_dict[f'{self.ns}.Eval.eval_inputs'] = self.input_selection_local_dv
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertListEqual(exec_eng.dm.get_value('doe.Eval.samples_df').columns.tolist(),
                             ['Eval.subprocess.Sellar_Problem.local_dv'])
        disc_dict[f'{self.ns}.Eval.eval_outputs'] = self.output_selection_obj_y1_y2
        disc_dict[f'{self.ns}.Eval.eval_inputs'] = self.input_selection_x_z
        exec_eng.load_study_from_input_dict(disc_dict)

        x_values = [array([9.379763880395856]), array([8.88644794300546]),
                    array([3.7137135749628882]), array([0.0417022004702574]), array([6.954954792150857])]
        z_values = [array([1.515949043849158, 5.6317362409322165]),
                    array([-1.1962705421254114, 6.523436208612142]),
                    array([-1.9947578026244557, 4.822570933860785]
                          ), array([1.7490668861813, 3.617234050834533]),
                    array([-9.316161097119341, 9.918161285133076])]

        samples_dict = {'x': x_values, 'z': z_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{self.ns}.Eval.samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {f'{self.ns}.x': array([1.]), f'{self.ns}.y_1': array([1.]), f'{self.ns}.y_2': array([1.]),
                       f'{self.ns}.z': array([1., 1.]), f'{self.ns}.Eval.subprocess.Sellar_Problem.local_dv': local_dv}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        eval_disc = exec_eng.dm.get_disciplines_with_name('doe.Eval')[0]

        eval_disc_samples = eval_disc.get_sosdisc_outputs(
            'samples_inputs_df')

        # check that the generated samples are the ones expected (custom sample
        # + reference value)
        expected_eval_disc_samples = pd.DataFrame(
            {'scenario': ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4', 'scenario_5', 'reference'],
             'x': x_values + [1.000000], 'z': z_values + [array([1.0, 1.0])]})
        assert_frame_equal(
            eval_disc_samples, expected_eval_disc_samples, check_dtype=False)

        # check that at the end of doe eval dm still contains initial
        # (reference) point
        self.assertEqual(exec_eng.dm.get_value('doe.x'), 1.0)
        self.assertEqual(exec_eng.dm.get_value(
            'doe.z').tolist(), array([1., 1.]).tolist())

    def test_6_doe_eval_design_space_normalisation(self):
        """ This tests aims at proving the ability of the
        doe factory to generate samples within the specified range
        """

        dspace_dict_x_eval = {'variable': ['x'],

                              'lower_bnd': [5.],
                              'upper_bnd': [11.]
                              }
        dspace_x_eval = pd.DataFrame(dspace_dict_x_eval)

        dspace_dict_eval = {'variable': ['x', 'z'],

                            'lower_bnd': [[-9.], [-10., 4.]],
                            'upper_bnd': [[150.], [10., 100.]]
                            }
        dspace_eval = pd.DataFrame(dspace_dict_eval)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # -- set up disciplines
        values_dict = {f'{self.ns}.x': array([1.]), f'{self.ns}.y_1': array([1.]), f'{self.ns}.y_2': array([1.]),
                       f'{self.ns}.z': array([1., 1.]), f'{self.ns}.DoEEval.subprocess.Sellar_Problem.local_dv': 10}
        exec_eng.load_study_from_input_dict(values_dict)

        # configure disciplines with the algo lhs and check that generated
        # samples are within default bounds
        disc_dict = {f'{self.ns}.DoEEval.sampling_algo': "lhs",
                     f'{self.ns}.DoEEval.eval_inputs': self.input_selection_x,
                     f'{self.ns}.DoEEval.eval_outputs': self.output_selection_obj_y1_y2,
                     'doe.DoEEval.algo_options': {'n_samples': 10, 'face': 'faced'}}

        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()
        # check that all generated samples (except the last one which is the
        # initial point) are within [0,10.] range
        generated_x = exec_eng.dm.get_value(
            'doe.DoEEval.samples_inputs_df')['x'].tolist()
        self.assertTrue(
            all(0 <= element[0] <= 10. for element in generated_x[:-1]))

        # trigger a reconfiguration after options and design space changes
        disc_dict = {'doe.DoEEval.design_space': dspace_x_eval}
        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()
        # check that all generated samples are within [5.,11.] range
        generated_x = exec_eng.dm.get_value(
            'doe.DoEEval.samples_inputs_df')['x'].tolist()
        self.assertTrue(
            all(5. <= element[0] <= 11. for element in generated_x[:-1]))

        # trigger a reconfiguration after algo name change
        disc_dict = {'doe.DoEEval.sampling_algo': "fullfact",
                     'doe.DoEEval.eval_outputs': self.output_selection_obj_y1_y2,
                     'doe.DoEEval.eval_inputs': self.input_selection_x_z,
                     'doe.DoEEval.design_space': dspace_eval}

        exec_eng.load_study_from_input_dict(disc_dict)
        disc_dict['doe.DoEEval.algo_options'] = {
            'n_samples': 10, 'face': 'faced'}
        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()
        generated_x = exec_eng.dm.get_value(
            'doe.DoEEval.samples_inputs_df')['x'].tolist()
        self.assertTrue(
            all(-9. <= element[0] <= 150. for element in generated_x[:-1]))

        generated_z = exec_eng.dm.get_value(
            'doe.DoEEval.samples_inputs_df')['z'].tolist()
        self.assertTrue(
            all(-10. <= element[0] <= 10. and 4. <= element[1] <= 100. for element in
                generated_z[:-1]))

    def _test_7_doe_eval_CustomDoE_reconfiguration_after_execution(self):
        """ This tests aims at proving the ability of the doe_eval to
        be reconfigured after execution
        """

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {f'{self.ns}.DoEEval.sampling_algo': "CustomDOE",
                     f'{self.ns}.DoEEval.eval_inputs': self.input_selection_local_dv_x,
                     f'{self.ns}.DoEEval.eval_outputs': self.output_selection_obj_y1_y2}
        # DoE inputs

        x_values = [array([9.379763880395856]), array([8.88644794300546]),
                    array([3.7137135749628882]), array([0.0417022004702574]), array([6.954954792150857])]
        local_dv_values = x_values

        samples_dict = {'x': x_values,
                        'DoEEval.subprocess.Sellar_Problem.local_dv': local_dv_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{self.ns}.DoEEval.samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {f'{self.ns}.x': array([1.]), f'{self.ns}.y_1': array([1.]), f'{self.ns}.y_2': array([1.]),
                       f'{self.ns}.z': array([1., 1.]), f'{self.ns}.DoEEval.subprocess.Sellar_Problem.local_dv': local_dv}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        doe_disc_y1 = doe_disc.get_sosdisc_outputs('y_1_dict')
        doe_disc_y2 = doe_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(doe_disc_samples), 6)
        self.assertEqual(len(doe_disc_obj), 6)
        self.assertEqual(len(doe_disc_y1), 6)
        self.assertEqual(len(doe_disc_y2), 6)

        disc_dict = {f'{self.ns}.DoEEval.eval_inputs': self.input_selection_x}
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertEqual(len(doe_disc_samples), 6)
        self.assertEqual(len(doe_disc_obj), 6)
        self.assertEqual(len(doe_disc_y1), 6)
        self.assertEqual(len(doe_disc_y2), 6)

    def test_7_2_Eval_User_defined_samples_reconfiguration_after_execution(self):
        """ This tests aims at proving the ability of the doe_eval to
        be reconfigured after execution
        """

        input_selection_x = {'selected_input': [False, True, False, False, False],
                             'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                           'y_2',
                                           'z']}
        self.input_selection_x = pd.DataFrame(input_selection_x)

        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        input_selection_local_dv_x = {'selected_input': [True, True, False, False, False],
                                      'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                    'y_2',
                                                    'z']}
        self.input_selection_local_dv_x = pd.DataFrame(
            input_selection_local_dv_x)

        input_selection_local_dv = {'selected_input': [True, False, False, False, False],
                                    'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                  'y_2',
                                                  'z']}
        self.input_selection_local_dv = pd.DataFrame(input_selection_local_dv)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {f'{self.ns}.Eval.eval_inputs': self.input_selection_local_dv_x,
                     f'{self.ns}.Eval.eval_outputs': self.output_selection_obj_y1_y2}
        # DoE inputs

        x_values = [array([9.379763880395856]), array([8.88644794300546]),
                    array([3.7137135749628882]), array([0.0417022004702574]), array([6.954954792150857])]
        local_dv_values = x_values

        samples_dict = {'x': x_values,
                        'Eval.subprocess.Sellar_Problem.local_dv': local_dv_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{self.ns}.Eval.samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {f'{self.ns}.x': array([1.]), f'{self.ns}.y_1': array([1.]), f'{self.ns}.y_2': array([1.]),
                       f'{self.ns}.z': array([1., 1.]),
                       f'{self.ns}.Eval.subprocess.Sellar_Problem.local_dv': local_dv}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ Eval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        eval_disc = exec_eng.dm.get_disciplines_with_name('doe.Eval')[0]

        eval_disc_samples = eval_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        eval_disc_obj = eval_disc.get_sosdisc_outputs('obj_dict')
        eval_disc_y1 = eval_disc.get_sosdisc_outputs('y_1_dict')
        eval_disc_y2 = eval_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(eval_disc_samples), 6)
        self.assertEqual(len(eval_disc_obj), 6)
        self.assertEqual(len(eval_disc_y1), 6)
        self.assertEqual(len(eval_disc_y2), 6)

        disc_dict = {f'{self.ns}.Eval.eval_inputs': self.input_selection_x}
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertEqual(len(eval_disc_samples), 6)
        self.assertEqual(len(eval_disc_obj), 6)
        self.assertEqual(len(eval_disc_y1), 6)
        self.assertEqual(len(eval_disc_y2), 6)
    def _test_8_doe_eval_parallel_execution(self):
        """ this tests evaluates the ability of the doe eval
        to store reference scenario value in dm when running
        in parallel
        """

        dspace_dict_x = {'variable': ['x'],

                         'lower_bnd': [0.],
                         'upper_bnd': [10.],

                         }
        dspace_x = pd.DataFrame(dspace_dict_x)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 10
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "lhs"
        disc_dict[f'{self.ns}.DoEEval.design_space'] = dspace_x
        disc_dict[f'{self.ns}.DoEEval.algo_options'] = {'n_samples': n_samples}
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {f'{self.ns}.x': array([1.]), f'{self.ns}.y_1': array([1.]), f'{self.ns}.y_2': array([1.]),
                       f'{self.ns}.z': array([1., 1.]), f'{self.ns}.DoEEval.subprocess.Sellar_Problem.local_dv': local_dv}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        doe_disc_y1 = doe_disc.get_sosdisc_outputs('y_1_dict')
        doe_disc_y2 = doe_disc.get_sosdisc_outputs('y_2_dict')

        for scenario, value in doe_disc_y1.items():
            print(scenario, ' : ', value)
        print(' the reference value of the doe is ' +
              str(exec_eng.dm.get_value('doe.x')))

        large_dspace_dict_x = {'variable': ['x'],

                               'lower_bnd': [0.],
                               'upper_bnd': [10000.],

                               }
        large_dspace_x = pd.DataFrame(large_dspace_dict_x)

        disc_dict = {}
        # DoE inputs
        n_samples = 10000
        n_processes = 10
        disc_dict[f'{self.ns}.DoEEval.design_space'] = large_dspace_x
        disc_dict[f'{self.ns}.DoEEval.algo_options'] = {
            'n_samples': n_samples, 'n_processes': n_processes}
        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()
        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        print(' number of samples retrieved after parallel execution ' +
              str(len(doe_disc_obj)))

    def test_9_doe_eval_with_2_outputs_with_the_same_name(self):
        """ Here we test that the doe displays properly 2 outputs
        with the same short name
        """

        dspace_dict = {'variable': ['x', 'DoEEval.subprocess.Disc1.a'],

                       'lower_bnd': [0., 50.],
                       'upper_bnd': [100., 200.],

                       }
        dspace = pd.DataFrame(dspace_dict)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory
        proc_name = "test_disc1_disc2_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Disc2',
                       '\t\t\t|_ Disc1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # -- set up disciplines
        private_values = {
            self.study_name + '.x': 10.,
            self.study_name + '.DoEEval.subprocess.Disc1.a': 5.,
            self.study_name + '.DoEEval.subprocess.Disc1.b': 25431.,
            self.study_name + '.y': 4.,
            self.study_name + '.DoEEval.subprocess.Disc2.constant': 3.1416,
            self.study_name + '.DoEEval.subprocess.Disc2.power': 2}
        exec_eng.load_study_from_input_dict(private_values)
        input_selection_x_a = {'selected_input': [True, True],
                               'full_name': ['x', 'DoEEval.subprocess.Disc1.a']}
        input_selection_x_a = pd.DataFrame(input_selection_x_a)

        output_selection_z_z = {'selected_output': [True, True],
                                'full_name': ['z', 'DoEEval.subprocess.Disc1.z']}
        output_selection_z_z = pd.DataFrame(output_selection_z_z)

        disc_dict = {f'{self.ns}.DoEEval.sampling_algo': "lhs", f'{self.ns}.DoEEval.eval_inputs': input_selection_x_a,
                     f'{self.ns}.DoEEval.eval_outputs': output_selection_z_z}

        exec_eng.load_study_from_input_dict(disc_dict)
        disc_dict = {'doe.DoEEval.algo_options': {'n_samples': 10, 'face': 'faced'},
                     'doe.DoEEval.design_space': dspace}

        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()
        self.assertEqual(len(exec_eng.dm.get_value(
            'doe.DoEEval.subprocess.Disc1.z_dict')), 11)
        self.assertEqual(len(exec_eng.dm.get_value('doe.z_dict')), 11)

        # Check coherence between ProxyCoupling of DoEEval and SoSMDAChain:
        self.assertEqual(set(exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].get_output_data_names()),
                         set(exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.get_output_data_names()))
        self.assertEqual(set(exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].get_input_data_names()),
                         set(exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.get_input_data_names()))
        # Test that the lower-level coupling does not crush inputs nor
        # numerical variables of its subprocess:
        self.assertIn('doe.DoEEval.subprocess.Disc2.cache_file_path',
                      exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].get_input_data_names())
        self.assertIn('doe.DoEEval.subprocess.Disc1.cache_file_path',
                      exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].get_input_data_names())
        # Check that the root process does not crush variables with the same
        # short name:
        for var in ['doe.z_dict', 'doe.DoEEval.subprocess.Disc1.z_dict', 'doe.z', 'doe.DoEEval.subprocess.Disc1.z']:
            self.assertIn(var, exec_eng.root_process.get_output_data_names())

    def test_10_warning_in_case_of_a_wrong_inputs_outputs_in_doe_eval(self):
        """ We check that a warning is displayed in doe eval in case the user
        sets a value for eval inputs or outputs which is not among the possible
        values.
        """

        wrong_input_selection_x = {'selected_input': [False, True, False, False, False],
                                   'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'debug_mode_sellar', 'y_1',
                                                 'y_2',
                                                 'z']}
        wrong_input_selection_x = pd.DataFrame(wrong_input_selection_x)

        wrong_output_selection_obj = {'selected_output': [False, False, True, False, False],
                                      'full_name': ['z', 'c_2', 'acceleration', 'y_1', 'y_2']}
        wrong_output_selection_obj = pd.DataFrame(wrong_output_selection_obj)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        exec_eng.logger.setLevel(logging.INFO)
        my_handler = UnitTestHandler()
        exec_eng.logger.addHandler(my_handler)

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines
        values_dict = {f'{self.ns}.x': 1., f'{self.ns}.y_1': 1., f'{self.ns}.y_2': 1., f'{self.ns}.z': array([1., 1.]),
                       f'{self.ns}.DoEEval.subprocess.Sellar_Problem.local_dv': 10}

        # configure disciplines with the algo lhs
        disc_dict = {f'{self.ns}.DoEEval.sampling_algo': "lhs",
                     f'{self.ns}.DoEEval.eval_inputs': wrong_input_selection_x,
                     f'{self.ns}.DoEEval.eval_outputs': wrong_output_selection_obj}

        disc_dict.update(values_dict)
        exec_eng.load_study_from_input_dict(disc_dict)

        msg_log_error_input_debug_mode = 'The input debug_mode_sellar in eval_inputs is not among possible values. Check if it is an input of the subprocess with the correct full name (without study name at the beginning) and within allowed types (int, array, float). Dynamic inputs might  not be created. '
        msg_log_error_output_z = 'The output z in eval_outputs is not among possible values. Check if it is an output of the subprocess with the correct full name (without study name at the beginning). Dynamic inputs might  not be created. '
        msg_log_error_acceleration = 'The output acceleration in eval_outputs is not among possible values. Check if it is an output of the subprocess with the correct full name (without study name at the beginning). Dynamic inputs might  not be created. '
        self.assertTrue(msg_log_error_input_debug_mode in my_handler.msg_list)
        self.assertTrue(msg_log_error_output_z in my_handler.msg_list)
        self.assertTrue(msg_log_error_acceleration in my_handler.msg_list)

    def test_11_doe_eval_execution_fullfact_in_case_of_not_standard_ns_doe_eval(self):
        """ We test that the ns_doe_eval can be set somewhere else as f'{self.ee.study_name}.DoEEval'
        """

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_ns_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 10
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "fullfact"
        disc_dict[f'{self.ns}.DoEEval.design_space'] = self.dspace_eval
        disc_dict[f'{self.ns}.DoEEval.algo_options'] = {
            'n_samples': n_samples, 'fake_option': 'fake_option'}
        disc_dict[f'{self.ns}.eval_inputs'] = self.input_selection_x_z
        disc_dict[f'{self.ns}.eval_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.subprocess.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')

        dimension = sum([len(sublist) if isinstance(
            sublist, list) else 1 for sublist in list(self.dspace_eval['lower_bnd'].values)])

        theoretical_fullfact_levels = int(n_samples ** (1.0 / dimension))

        theoretical_fullfact_samples = theoretical_fullfact_levels ** dimension
        self.assertEqual(len(doe_disc_samples),
                         theoretical_fullfact_samples + 1)

    def _test_12_doe_eval_CustomDoE_non_alpha_order(self):
        """ this test checks that the doe_eval delivers expected outputs in case of invertion of x, z in eval_inputs
        It is a non regression test
        """
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        input_selection_z_x = {'selected_input': [False, True, True, False, False],
                               'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'z', 'x', 'y_1',
                                             'y_2']}
        input_selection_z_x = pd.DataFrame(input_selection_z_x)

        disc_dict = {f'{self.ns}.DoEEval.sampling_algo': "CustomDOE",
                     f'{self.ns}.DoEEval.eval_inputs': input_selection_z_x,
                     f'{self.ns}.DoEEval.eval_outputs': self.output_selection_obj_y1_y2}
        # DoE inputs

        x_values = [array([9.379763880395856]), array([8.88644794300546]),
                    array([3.7137135749628882]), array([0.0417022004702574]), array([6.954954792150857])]
        z_values = [array([1.515949043849158, 5.6317362409322165]),
                    array([-1.1962705421254114, 6.523436208612142]),
                    array([-1.9947578026244557, 4.822570933860785]
                          ), array([1.7490668861813, 3.617234050834533]),
                    array([-9.316161097119341, 9.918161285133076])]
        wrong_values = 5 * [0.0]

        #samples_dict = {'x': x_values, 'z': z_values,'wrong_values':wrong_values}
        samples_dict = {'z': z_values, 'x': x_values,
                        'wrong_values': wrong_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{self.ns}.DoEEval.samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {f'{self.ns}.x': array([1.]), f'{self.ns}.y_1': array([1.]), f'{self.ns}.y_2': array([1.]),
                       f'{self.ns}.z': array([1., 1.]), f'{self.ns}.DoEEval.subprocess.Sellar_Problem.local_dv': local_dv}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        doe_disc_y1 = doe_disc.get_sosdisc_outputs('y_1_dict')
        doe_disc_y2 = doe_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(doe_disc_samples), 6)
        self.assertEqual(len(doe_disc_obj), 6)
        doe_disc_y1_ref = {'scenario_1': array([15.102817691025274]), 'scenario_2': array([15.000894464408367]),
                           'scenario_3': array([11.278122259980103]), 'scenario_4': array([5.1893098993071565]),
                           'scenario_5': array([101.52834810032466]), 'reference': array([2.29689011157193])}
        doe_disc_y2_ref = {'scenario_1': array([11.033919669249876]), 'scenario_2': array([9.200264485831308]),
                           'scenario_3': array([6.186104699873589]), 'scenario_4': array([7.644306621667905]),
                           'scenario_5': array([10.67812782219566]), 'reference': array([3.515549442140351])}

        for key, key_ref in zip(doe_disc_y1, doe_disc_y1_ref):
            self.assertEqual(key, key_ref)
            self.assertAlmostEqual(
                doe_disc_y1[key][0], doe_disc_y1_ref[key_ref][0])
        for key, key_ref in zip(doe_disc_y2, doe_disc_y2_ref):
            self.assertEqual(key, key_ref)
            self.assertAlmostEqual(
                doe_disc_y2[key][0], doe_disc_y2_ref[key_ref][0])

    def test_12_2_Eval_User_Defined_samples_non_alpha(self):
        """
        This test checks that the custom samples applied to an Eval driver delivers expected outputs
        It is a non regression test
        """
        study_name = 'root'
        ns = study_name

        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_eval"
        eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                        mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            eval_builder)

        exec_eng.configure()

        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['Eval.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
                                      'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj_y1_y2 = pd.DataFrame(
            output_selection_obj_y1_y2)

        # -- set up disciplines in Scenario
        disc_dict = {f'{ns}.Eval.eval_inputs': self.input_selection_x_z,
                     f'{ns}.Eval.eval_outputs': self.output_selection_obj_y1_y2}
        # Eval inputs

        x_values = [array([9.379763880395856]), array([8.88644794300546]),
                    array([3.7137135749628882]), array([0.0417022004702574]), array([6.954954792150857])]
        z_values = [array([1.515949043849158, 5.6317362409322165]),
                    array([-1.1962705421254114, 6.523436208612142]),
                    array([-1.9947578026244557, 4.822570933860785]
                          ), array([1.7490668861813, 3.617234050834533]),
                    array([-9.316161097119341, 9.918161285133076])]

        wrong_values = 5 * [0.0]

        # samples_dict = {'x': x_values, 'z': z_values,'wrong_values':wrong_values}
        samples_dict = {'z': z_values, 'x': x_values,
                        'wrong_values': wrong_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{ns}.Eval.samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {f'{ns}.x': array([1.]), f'{ns}.y_1': array([1.]), f'{ns}.y_2': array([1.]),
                       f'{ns}.z': array([1., 1.]),
                       f'{ns}.Eval.subprocess.Sellar_Problem.local_dv': local_dv}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {ns}',
                       '|_ root',
                       f'\t|_ Eval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        root_outputs = exec_eng.root_process.get_output_data_names()
        self.assertIn('root.obj_dict', root_outputs)
        self.assertIn('root.y_1_dict', root_outputs)
        self.assertIn('root.y_2_dict', root_outputs)

        # doe_disc = exec_eng.dm.get_disciplines_with_name(f'{ns}.Eval')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp
        doe_disc = exec_eng.dm.get_disciplines_with_name(f'{ns}.Eval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        doe_disc_y1 = doe_disc.get_sosdisc_outputs('y_1_dict')
        doe_disc_y2 = doe_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(doe_disc_samples), 6)
        self.assertEqual(len(doe_disc_obj), 6)
        reference_dict_doe_disc_y1 = {'scenario_1': array([15.102817691025274]),
                                      'scenario_2': array([15.000894464408367]),
                                      'scenario_3': array([11.278122259980103]),
                                      'scenario_4': array([5.1893098993071565]),
                                      'scenario_5': array([101.52834810032466]), 'reference': array([2.29689011157193])}
        reference_dict_doe_disc_y2 = {'scenario_1': array([11.033919669249876]),
                                      'scenario_2': array([9.200264485831308]),
                                      'scenario_3': array([6.186104699873589]),
                                      'scenario_4': array([7.644306621667905]),
                                      'scenario_5': array([10.67812782219566]), 'reference': array([3.515549442140351])}
        for key in doe_disc_y1.keys():
            self.assertAlmostEqual(
                doe_disc_y1[key][0], reference_dict_doe_disc_y1[key][0])
        for key in doe_disc_y2.keys():
            self.assertAlmostEqual(
                doe_disc_y2[key][0], reference_dict_doe_disc_y2[key][0])

    def test_13_sameusecase_name_as_doe_eval(self):
        """ We test that the number of samples generated by the fullfact algorithm is the theoretical expected number
        Pay attention to the fact that an additional sample (the reference one ) is added
        """
        same_usecase_name = 'DoEEval'
        exec_eng = ExecutionEngine(same_usecase_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 10
        disc_dict[f'{same_usecase_name}.DoEEval.sampling_algo'] = "fullfact"
        disc_dict[f'{same_usecase_name}.DoEEval.design_space'] = self.dspace_eval
        disc_dict[f'{same_usecase_name}.DoEEval.algo_options'] = {
            'n_samples': n_samples, 'fake_option': 'fake_option'}
        disc_dict[f'{same_usecase_name}.DoEEval.eval_inputs'] = self.input_selection_x_z
        disc_dict[f'{same_usecase_name}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{same_usecase_name}.x'] = array([1.])
        values_dict[f'{same_usecase_name}.y_1'] = array([1.])
        values_dict[f'{same_usecase_name}.y_2'] = array([1.])
        values_dict[f'{same_usecase_name}.z'] = array([1., 1.])
        values_dict[f'{same_usecase_name}.DoEEval.subprocess.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {same_usecase_name}',
                       f'|_ {same_usecase_name}',
                       f'\t|_ DoEEval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name(
            f'{same_usecase_name}.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')

        dimension = sum([len(sublist) if isinstance(
            sublist, list) else 1 for sublist in list(self.dspace_eval['lower_bnd'].values)])

        theoretical_fullfact_levels = int(n_samples ** (1.0 / dimension))

        theoretical_fullfact_samples = theoretical_fullfact_levels ** dimension
        self.assertEqual(len(doe_disc_samples),
                         theoretical_fullfact_samples + 1)

    def test_14_doe_eval_of_single_sub_discipline(self):
        """ Here we test a DoEEval process on a single sub-discipline so that there is no ProxyCoupling built in node.
        """

        dspace_dict = {'variable': ['x'],

                       'lower_bnd': [0.],
                       'upper_bnd': [100.],

                       }
        dspace = pd.DataFrame(dspace_dict)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory
        proc_name = "test_disc1_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ Disc1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        assert not exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].is_sos_coupling

        # -- set up disciplines
        private_values = {
            self.study_name + '.x': array([10.]),
            self.study_name + '.DoEEval.Disc1.a': array([5.]),
            self.study_name + '.DoEEval.Disc1.b': array([25431.]),
            self.study_name + '.y': array([4.])}
        exec_eng.load_study_from_input_dict(private_values)
        input_selection_a = {'selected_input': [False, True, False],
                             'full_name': ['x', 'DoEEval.Disc1.a', 'DoEEval.Disc1.b']}
        input_selection_a = pd.DataFrame(input_selection_a)

        output_selection_ind = {'selected_output': [False, True],
                                'full_name': ['y', 'DoEEval.Disc1.indicator']}
        output_selection_ind = pd.DataFrame(output_selection_ind)

        disc_dict = {f'{self.ns}.DoEEval.sampling_algo': "lhs",
                     f'{self.ns}.DoEEval.eval_inputs': input_selection_a,
                     f'{self.ns}.DoEEval.eval_outputs': output_selection_ind}

        exec_eng.load_study_from_input_dict(disc_dict)
        disc_dict = {'doe.DoEEval.algo_options': {'n_samples': 10, 'face': 'faced'},
                     'doe.DoEEval.design_space': dspace}

        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()

        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        doe_disc_ind = doe_disc.get_sosdisc_outputs(
            'DoEEval.Disc1.indicator_dict')

        self.assertEqual(len(doe_disc_ind), 11)
        i = 0
        for key in doe_disc_ind.keys():
            self.assertAlmostEqual(doe_disc_ind[key],
                                   private_values[self.study_name + '.DoEEval.Disc1.b'] * doe_disc_samples['DoEEval.Disc1.a'][i][0])
            i += 1

    def test_15_DoE_OT_FACTORIAL_Eval(self):
        """

        """

        input_selection_a = {'selected_input': [False, True, False],
                             'full_name': ['x', 'Eval.Disc1.a', 'Eval.Disc1.b']}
        input_selection_a = pd.DataFrame(input_selection_a)

        output_selection_ind = {'selected_output': [False, True],
                                'full_name': ['y', 'Eval.Disc1.indicator']}
        output_selection_ind = pd.DataFrame(output_selection_ind)

        dspace_dict_a = {'variable': ['Eval.Disc1.a'],

                         'lower_bnd': [0.],
                         'upper_bnd': [1.],

                         }
        dspace_a = pd.DataFrame(dspace_dict_a)

        study_name = 'doe'
        ns = study_name

        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory
        proc_name = "test_disc1_doe_eval_separated"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {ns}',
                       '|_ doe',
                       f'\t|_ DoE',
                       '\t|_ Eval',
                       '\t\t|_ Disc1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        assert not exec_eng.root_process.proxy_disciplines[1].proxy_disciplines[0].is_sos_coupling

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 20
        levels = [0.25, 0.5, 0.75]
        centers = [5]
        disc_dict[f'{ns}.DoE.sampling_algo'] = 'OT_FACTORIAL'
        disc_dict[f'{ns}.DoE.design_space'] = dspace_a
        disc_dict[f'{ns}.DoE.algo_options'] = {'n_samples': n_samples, 'levels': levels, 'centers': centers}
        disc_dict[f'{ns}.eval_inputs'] = input_selection_a

        # Eval inputs
        # disc_dict[f'{self.ns}.eval_inputs'] = disc_dict[f'{self.ns}.eval_inputs']
        disc_dict[f'{ns}.eval_outputs'] = output_selection_ind
        exec_eng.load_study_from_input_dict(disc_dict)

        # -- Discipline inputs
        private_values = {
            f'{ns}.x': array([10.]),
            f'{ns}.Eval.Disc1.a': array([0.5]),
            f'{ns}.Eval.Disc1.b': array([25431.]),
            f'{ns}.y': array([4.]),
            f'{ns}.Eval.Disc1.indicator': array([53.])}
        exec_eng.load_study_from_input_dict(private_values)

        exec_eng.execute()

        root_outputs = exec_eng.root_process.get_output_data_names()
        self.assertIn('doe.Eval.Disc1.indicator_dict', root_outputs)

        eval_disc = exec_eng.dm.get_disciplines_with_name(
            study_name + '.Eval')[0]

        eval_disc_samples = eval_disc.get_sosdisc_outputs(
            'samples_inputs_df')

        eval_disc_ind = eval_disc.get_sosdisc_outputs(
            'Eval.Disc1.indicator_dict')

        i = 0
        for key in eval_disc_ind.keys():
            self.assertTrue(0. <= eval_disc_samples['Eval.Disc1.a'][i] <= 1.)
            self.assertAlmostEqual(eval_disc_ind[key],
                                   private_values[f'{ns}.Eval.Disc1.b'] * eval_disc_samples['Eval.Disc1.a'][i][0])
            i += 1

if '__main__' == __name__:
    cls = TestSoSDOEScenario()
    cls.setUp()
    cls._test_8_doe_eval_parallel_execution()
