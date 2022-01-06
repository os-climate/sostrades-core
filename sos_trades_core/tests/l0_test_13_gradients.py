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
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from sos_trades_core.tools.grad_solvers.validgrad.FDGradient import FDGradient
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from gemseo.core.discipline import MDODiscipline


class TestGradients(unittest.TestCase):
    """
    Gradients test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'EETests'
        self.exec_eng = ExecutionEngine(self.name)
        self.repo = 'sos_trades_core.sos_processes.test'
        self.sub_proc = 'test_disc1_disc2_coupling'

    def demo_func(self, x_in):
        a = x_in[0]
        x = x_in[1]
        b = x_in[2]
        cst = x_in[3]
        power = x_in[4]

        func = cst + (a * x + b)**power

        return func

    def test_01_gradient_analysis_configure(self):

        ns_dict = {'ns_ac': f'{self.name}.AC'}
        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        builder_list = self.exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                      mod_id=self.sub_proc)

        gradient_builder = self.exec_eng.factory.create_evaluator_builder(
            'GA', 'gradient', builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            gradient_builder)

        self.exec_eng.configure()

        print('\nTreeView display 2')
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = ['Nodes representation for Treeview EETests',
                       '|_ EETests',
                       '\t|_ GA',
                       '\t\t|_ Disc1',
                       '\t\t|_ Disc2']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_02_gradient_analysis_execute(self):

        builder_list = self.exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                      mod_id=self.sub_proc)

        gradient_builder = self.exec_eng.factory.create_evaluator_builder(
            'GA', 'gradient', builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            gradient_builder)

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        #[a,x,b,cst,power]
        x0 = np.array([3., 2., 10., -10., 2])

        values_dict = {}
        values_dict['EETests.GA.eval_inputs'] = ['a', 'x',
                                                 'b', 'constant']
        values_dict['EETests.GA.eval_outputs'] = ['y', 'z']
        values_dict['EETests.GA.grad_method'] = 'Complex Step'
        values_dict['EETests.GA.Disc1.a'] = x0[0]
        values_dict['EETests.x'] = x0[1]
        values_dict['EETests.GA.Disc1.b'] = x0[2]
        values_dict['EETests.GA.Disc2.constant'] = x0[3]
        values_dict['EETests.GA.Disc2.power'] = int(x0[4])

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

        gradients_output = self.exec_eng.dm.get_value(
            'EETests.GA.gradient_outputs')

        self.exec_eng.execute()

        gradients_output_2 = self.exec_eng.dm.get_value(
            'EETests.GA.gradient_outputs')

        self.assertDictEqual(gradients_output, gradients_output_2,
                             'Two execute in a raw do not give the same result ')

        #-- Ref gradient computation
        err_tol = 4
        eps = round(0.1 ** err_tol, err_tol + 1)
        grad_eval = FDGradient(2, self.demo_func, fd_step=eps)
        grad_eval.set_multi_proc(False)
        outputs_grad = grad_eval.grad_f(x0)

        print('z ref_gradient =', outputs_grad)

        key_list = ['EETests.z vs EETests.GA.Disc1.a',
                    'EETests.z vs EETests.x',
                    'EETests.z vs EETests.GA.Disc1.b',
                    'EETests.z vs EETests.GA.Disc2.constant']

        gradient_output_list = [
            gradients_output[key] for key in key_list]
        assert_array_almost_equal(
            gradient_output_list, outputs_grad[:-1], decimal=err_tol)

        # Check on graphs :
#         disc = self.exec_eng.dm.get_disciplines_with_name(
#             'EETests.GA')[0]
#         filters = disc.get_chart_filter_list()
#         graph_list = disc.get_post_processing_list(filters)

    def test_03_gradients_disc1_gems(self):

        ns_dict = {'ns_ac': f'{self.name}'}
        self.exec_eng.ns_manager.add_ns_def(ns_dict)
        # macro_economics
        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc1.Disc1'
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', mod_list)
        self.exec_eng.factory.set_builders_to_coupling_builder(disc1_builder)

        self.exec_eng.configure()
        data_dict = {}
        data_dict['EETests.x'] = np.array([5.])
        data_dict['EETests.Disc1.a'] = np.array([5.])
        data_dict['EETests.Disc1.b'] = np.array([20.])
        self.exec_eng.load_study_from_input_dict(data_dict)

        disc = self.exec_eng.root_process.sos_disciplines[0]

        #-- run gradient with complex step and finite differences
        dy_dx_ref = disc.get_sosdisc_inputs('a')
        for approx_method in MDODiscipline.APPROX_MODES:
            print("\t Test with approximation mode ", approx_method)
            disc.linearization_mode = approx_method
            jac = disc.linearize(data_dict, force_all=True)
            dy_dx = jac['EETests.y']['EETests.x'][0]
            print(jac)
            msg = "Gradient computation error with method " + approx_method
            assert_array_almost_equal(dy_dx, dy_dx_ref, err_msg=msg, decimal=8)

    def test_04_set_grad_possible_values(self):

        builder_list = self.exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                      mod_id=self.sub_proc)

        gradient_builder = self.exec_eng.factory.create_evaluator_builder(
            'GA', 'gradient', builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            gradient_builder)

        self.exec_eng.configure()

        soseval = self.exec_eng.root_process.sos_disciplines[1]
        soseval.set_eval_possible_values()

        id_map = self.exec_eng.dm.data_id_map[self.name + '.GA.eval_inputs']
        possible_in_values = self.exec_eng.dm.data_dict[id_map
                                                        ]['possible_values']
        id_map = self.exec_eng.dm.data_id_map[self.name + '.GA.eval_outputs']
        possible_out_values = self.exec_eng.dm.data_dict[id_map
                                                         ]['possible_values']

        possible_in_values_ref = ['b', 'a', 'constant', 'y', 'x']
        possible_out_values_ref = ['y', 'z', 'indicator', 'residuals_history']

        self.assertListEqual(sorted(possible_in_values), sorted(possible_in_values_ref),
                             'The list of possible values for eval inputs is not correct')

        self.assertListEqual(sorted(possible_out_values), sorted(possible_out_values_ref),
                             'The list of possible values for eval outputs is not correct')

    def test_05_sensitivity_analysis_execute(self):
        '''
            Compare a sensitivity analysis with the same execution
        '''
        builder_list = self.exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                      mod_id=self.sub_proc)

        sa_builder = self.exec_eng.factory.create_evaluator_builder(
            'SA', 'sensitivity', builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(sa_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        #[a,x,b,cst,power]
        x0 = np.array([3., 2., 10., -10., 2])

        values_dict = {}
        values_dict['EETests.SA.eval_inputs'] = ['a', 'x',
                                                 'b', 'constant']
        values_dict['EETests.SA.eval_outputs'] = ['y', 'z']
        values_dict['EETests.SA.variation_list'] = ['+/-5%']
        values_dict['EETests.SA.Disc1.a'] = x0[0]
        values_dict['EETests.x'] = x0[1]
        values_dict['EETests.SA.Disc1.b'] = x0[2]
        values_dict['EETests.SA.Disc2.constant'] = x0[3]
        values_dict['EETests.SA.Disc2.power'] = int(x0[4])

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

        sensitivity_output = self.exec_eng.dm.get_value(
            'EETests.SA.sensitivity_outputs')

        self.exec_eng.execute()

        sensitivity_output_2 = self.exec_eng.dm.get_value(
            'EETests.SA.sensitivity_outputs')

        self.assertDictEqual(sensitivity_output, sensitivity_output_2,
                             'Two execute in a raw do not give the same result ')

        print(sensitivity_output['+5.0%'])
        sensitivity_output_ref_5_percent = {'EETests.y vs EETests.SA.Disc1.a': 0.3000000000000007,
                                            'EETests.z vs EETests.SA.Disc1.a': 9.689999999999998,
                                            'EETests.y vs EETests.x': 0.3000000000000007,
                                            'EETests.z vs EETests.x': 9.689999999999998,
                                            'EETests.y vs EETests.SA.Disc1.b': 0.5,
                                            'EETests.z vs EETests.SA.Disc1.b': 16.25,
                                            'EETests.y vs EETests.SA.Disc2.constant': 0.0,
                                            'EETests.z vs EETests.SA.Disc2.constant': -0.5}

        self.assertDictEqual(sensitivity_output_ref_5_percent, sensitivity_output['+5.0%'],
                             'The comparison with the reference is not correct')
        # Check on graphs :
#         disc = self.exec_eng.dm.get_disciplines_with_name(
#             'EETests.SA')[0]
#         filters = disc.get_chart_filter_list()
#         graph_list = disc.get_post_processing_list(filters)
#         for graph in graph_list:
#             graph.to_plotly().show()

    def test_06_hessian_with_double_gradient(self):

        base_path = 'sos_trades_core.sos_wrapping.test_discs'
        mod_path = f'{base_path}.disc_hessian.DiscHessian'
        builder_list = self.exec_eng.factory.get_builder_from_module(
            'DiscHessian', mod_path)

        self.exec_eng.ns_manager.add_ns('ns_grad1', f'{self.name}.GA2.GA')

        gradient_builder1 = self.exec_eng.factory.create_evaluator_builder(
            'GA', 'gradient', builder_list)

        mod_path = f'{base_path}.disc_convertgrad.DiscConvertGrad'
        disc_convert_grad = self.exec_eng.factory.get_builder_from_module(
            'DiscConvertGrad', mod_path)
        gradient_builder2 = self.exec_eng.factory.create_evaluator_builder(
            'GA2', 'gradient', [gradient_builder1, disc_convert_grad])

        self.exec_eng.factory.set_builders_to_coupling_builder(
            gradient_builder2)

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        print(self.exec_eng.dm.disciplines_id_map.keys())

        x = 2.0
        y = 3.0
        ax2 = 4.0
        by2 = 5.0
        cx = 6.0
        dy = 7.0
        exy = 12.0
        values_dict = {}
        values_dict['EETests.GA2.GA.eval_inputs'] = ['x', 'y']
        values_dict['EETests.GA2.GA.eval_outputs'] = ['z']
        values_dict['EETests.GA2.GA.grad_method'] = '2nd order FD'
        values_dict['EETests.GA2.eval_inputs'] = ['x', 'y']
        values_dict['EETests.GA2.eval_outputs'] = ['dzdx', 'dzdy']
        values_dict['EETests.GA2.grad_method'] = '2nd order FD'
        values_dict['EETests.GA2.GA.DiscHessian.x'] = x
        values_dict['EETests.GA2.GA.DiscHessian.y'] = y
        values_dict['EETests.GA2.GA.DiscHessian.ax2'] = ax2
        values_dict['EETests.GA2.GA.DiscHessian.by2'] = by2
        values_dict['EETests.GA2.GA.DiscHessian.cx'] = cx
        values_dict['EETests.GA2.GA.DiscHessian.dy'] = dy
        values_dict['EETests.GA2.GA.DiscHessian.exy'] = exy

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

        gradients_output_ga2 = self.exec_eng.dm.get_value(
            'EETests.GA2.gradient_outputs')

        dzdx = self.exec_eng.dm.get_value(
            'EETests.GA2.DiscConvertGrad.dzdx')

        self.assertAlmostEqual(dzdx,  2 * ax2 * x + cx + exy * y, delta=2.e-3)
        dzdy = self.exec_eng.dm.get_value(
            'EETests.GA2.DiscConvertGrad.dzdy')
        self.assertAlmostEqual(dzdy,  2 * by2 * y + dy + exy * x, delta=2.e-3)

        dzdy2 = 2 * by2
        dzdx2 = 2 * ax2
        dzdxy = exy
        ref_output_second_gradient = {'EETests.GA2.DiscConvertGrad.dzdx vs EETests.GA2.GA.DiscHessian.x': dzdx2,
                                      'EETests.GA2.DiscConvertGrad.dzdy vs EETests.GA2.GA.DiscHessian.x': dzdxy,
                                      'EETests.GA2.DiscConvertGrad.dzdx vs EETests.GA2.GA.DiscHessian.y': dzdxy,
                                      'EETests.GA2.DiscConvertGrad.dzdy vs EETests.GA2.GA.DiscHessian.y': dzdy2}

        for key_ref, value_grad in ref_output_second_gradient.items():
            self.assertAlmostEqual(
                gradients_output_ga2[key_ref], value_grad, delta=2.e-3)

    def test_07_sensitivity_analysis_on_scatter(self):
        '''
            Compare a sensitivity analysis with the same execution
        '''
        builder_list = self.exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                      mod_id='test_coupling_of_scatter')

        sa_builder = self.exec_eng.factory.create_evaluator_builder(
            'SA', 'sensitivity', builder_list)
#         self.exec_eng.ns_manager.update_all_shared_namespaces_by_name(
#             'SA', 'ns_barrierr', 'EETests')
        self.exec_eng.factory.set_builders_to_coupling_builder(sa_builder)
        for ns in self.exec_eng.ns_manager.ns_list:
            ns.update_value(f'{ns.value}.SA')

        self.exec_eng.configure()

        self.exec_eng.display_treeview_nodes()

        #[a,x,b,cst,power]
        x0 = np.array([3., 2., 10., -10., 2])
        name_list = ['name_1', 'name_2']
        values_dict = {}
        values_dict['EETests.SA.name_list'] = name_list

        self.exec_eng.load_study_from_input_dict(values_dict)

        self.exec_eng.display_treeview_nodes()
        values_dict = {}
        values_dict['EETests.SA.eval_inputs'] = ['a', 'x',
                                                 'b', 'constant']
        values_dict['EETests.SA.eval_outputs'] = ['y', 'z']
        values_dict['EETests.SA.variation_list'] = ['+/-5%']

        for key in self.exec_eng.dm.data_id_map:
            print(key)

        for aircraft in name_list:
            values_dict[f'EETests.SA.Disc1.{aircraft}.a'] = x0[0]
            values_dict[f'EETests.SA.{aircraft}.x'] = x0[1]
            values_dict[f'EETests.SA.Disc1.{aircraft}.b'] = x0[2]
            values_dict[f'EETests.SA.Disc2.{aircraft}.constant'] = x0[3]
            values_dict[f'EETests.SA.Disc2.{aircraft}.power'] = int(x0[4])

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

        sensitivity_output = self.exec_eng.dm.get_value(
            'EETests.SA.sensitivity_outputs')

        self.exec_eng.execute()

        sensitivity_output_2 = self.exec_eng.dm.get_value(
            'EETests.SA.sensitivity_outputs')

        self.assertDictEqual(sensitivity_output, sensitivity_output_2,
                             'Two execute in a raw do not give the same result ')

        print(sensitivity_output['+5.0%'])

        sensitivity_output_ref_5_percent = {'EETests.SA.name_1.y vs EETests.SA.Disc1.name_1.a': 0.3000000000000007,
                                            'EETests.SA.name_2.y vs EETests.SA.Disc1.name_1.a': 0.0,
                                            'EETests.SA.name_1.z vs EETests.SA.Disc1.name_1.a': 9.689999999999998,
                                            'EETests.SA.name_2.z vs EETests.SA.Disc1.name_1.a': 0.0,
                                            'EETests.SA.name_1.y vs EETests.SA.Disc1.name_2.a': 0.0,
                                            'EETests.SA.name_2.y vs EETests.SA.Disc1.name_2.a': 0.3000000000000007,
                                            'EETests.SA.name_1.z vs EETests.SA.Disc1.name_2.a': 0.0,
                                            'EETests.SA.name_2.z vs EETests.SA.Disc1.name_2.a': 9.689999999999998,
                                            'EETests.SA.name_1.y vs EETests.SA.name_1.x': 0.3000000000000007,
                                            'EETests.SA.name_2.y vs EETests.SA.name_1.x': 0.0,
                                            'EETests.SA.name_1.z vs EETests.SA.name_1.x': 9.689999999999998,
                                            'EETests.SA.name_2.z vs EETests.SA.name_1.x': 0.0,
                                            'EETests.SA.name_1.y vs EETests.SA.name_2.x': 0.0,
                                            'EETests.SA.name_2.y vs EETests.SA.name_2.x': 0.3000000000000007,
                                            'EETests.SA.name_1.z vs EETests.SA.name_2.x': 0.0,
                                            'EETests.SA.name_2.z vs EETests.SA.name_2.x': 9.689999999999998,
                                            'EETests.SA.name_1.y vs EETests.SA.Disc1.name_1.b': 0.5,
                                            'EETests.SA.name_2.y vs EETests.SA.Disc1.name_1.b': 0.0,
                                            'EETests.SA.name_1.z vs EETests.SA.Disc1.name_1.b': 16.25,
                                            'EETests.SA.name_2.z vs EETests.SA.Disc1.name_1.b': 0.0,
                                            'EETests.SA.name_1.y vs EETests.SA.Disc1.name_2.b': 0.0,
                                            'EETests.SA.name_2.y vs EETests.SA.Disc1.name_2.b': 0.5,
                                            'EETests.SA.name_1.z vs EETests.SA.Disc1.name_2.b': 0.0,
                                            'EETests.SA.name_2.z vs EETests.SA.Disc1.name_2.b': 16.25,
                                            'EETests.SA.name_1.y vs EETests.SA.Disc2.name_1.constant': 0.0,
                                            'EETests.SA.name_2.y vs EETests.SA.Disc2.name_1.constant': 0.0,
                                            'EETests.SA.name_1.z vs EETests.SA.Disc2.name_1.constant': -0.5,
                                            'EETests.SA.name_2.z vs EETests.SA.Disc2.name_1.constant': 0.0,
                                            'EETests.SA.name_1.y vs EETests.SA.Disc2.name_2.constant': 0.0,
                                            'EETests.SA.name_2.y vs EETests.SA.Disc2.name_2.constant': 0.0,
                                            'EETests.SA.name_1.z vs EETests.SA.Disc2.name_2.constant': 0.0,
                                            'EETests.SA.name_2.z vs EETests.SA.Disc2.name_2.constant': -0.5}

        self.assertDictEqual(sensitivity_output_ref_5_percent, sensitivity_output['+5.0%'],
                             'The comparison with the reference is not correct')

        # Check on graphs :
        disc = self.exec_eng.dm.get_disciplines_with_name(
            'EETests.SA')[0]
        filters = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filters)
#         for graph in graph_list:
#             graph.to_plotly().show()

    def test_08_form_analysis_on_scatter(self):
        '''
            Compare a sensitivity analysis with the same execution
        '''
        builder_list = self.exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                      mod_id='test_coupling_of_scatter')

        sa_builder = self.exec_eng.factory.create_evaluator_builder(
            'FORM', 'FORM', builder_list)
#         self.exec_eng.ns_manager.update_all_shared_namespaces_by_name(
#             'SA', 'ns_barrierr', 'EETests')
        self.exec_eng.factory.set_builders_to_coupling_builder(sa_builder)
        for ns in self.exec_eng.ns_manager.ns_list:
            ns.update_value(f'{ns.value}.FORM')

        self.exec_eng.configure()

        self.exec_eng.display_treeview_nodes()

        #[a,x,b,cst,power]
        x0 = np.array([3., 2., 10., -10., 2])
        name_list = ['name_1', 'name_2']
        values_dict = {}
        values_dict['EETests.FORM.name_list'] = name_list

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.display_treeview_nodes()
        values_dict = {}
        values_dict['EETests.FORM.eval_inputs'] = ['a', 'x',
                                                   'b', 'constant']
        values_dict['EETests.FORM.eval_outputs'] = ['y', 'z']
        values_dict['EETests.FORM.variation_list'] = ['+/-5%']
        values_dict['EETests.FORM.grad_method'] = '2nd order FD'
        for key in self.exec_eng.dm.data_id_map:
            print(key)

        for aircraft in name_list:
            values_dict[f'EETests.FORM.Disc1.{aircraft}.a'] = x0[0]
            values_dict[f'EETests.FORM.{aircraft}.x'] = x0[1]
            values_dict[f'EETests.FORM.Disc1.{aircraft}.b'] = x0[2]
            values_dict[f'EETests.FORM.Disc2.{aircraft}.constant'] = x0[3]
            values_dict[f'EETests.FORM.Disc2.{aircraft}.power'] = int(x0[4])

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

        form_output = self.exec_eng.dm.get_value(
            'EETests.FORM.FORM_outputs')

        self.exec_eng.execute()

        form_output_2 = self.exec_eng.dm.get_value(
            'EETests.FORM.FORM_outputs')

        self.assertDictEqual(form_output, form_output_2,
                             'Two execute in a raw do not give the same result ')

        print(form_output['5.0%'])

        form_output_ref_5_percent = {'EETests.FORM.name_1.y vs EETests.FORM.Disc1.name_1.a': 0.2999999999993009,
                                     'EETests.FORM.name_2.y vs EETests.FORM.Disc1.name_1.a': 0.0,
                                     'EETests.FORM.name_1.z vs EETests.FORM.Disc1.name_1.a': 0.0,
                                     'EETests.FORM.name_2.z vs EETests.FORM.Disc1.name_1.a': 0.0,
                                     'EETests.FORM.name_1.y vs EETests.FORM.Disc1.name_2.a': 0.0,
                                     'EETests.FORM.name_2.y vs EETests.FORM.Disc1.name_2.a': 0.2999999999993009,
                                     'EETests.FORM.name_1.z vs EETests.FORM.Disc1.name_2.a': 0.2999999999993009,
                                     'EETests.FORM.name_2.z vs EETests.FORM.Disc1.name_2.a': 0.2999999999993009,
                                     'EETests.FORM.name_1.y vs EETests.FORM.name_1.x': 0.3000000000010772,
                                     'EETests.FORM.name_2.y vs EETests.FORM.name_1.x': 0.0,
                                     'EETests.FORM.name_1.z vs EETests.FORM.name_1.x': 0.0,
                                     'EETests.FORM.name_2.z vs EETests.FORM.name_1.x': 0.0,
                                     'EETests.FORM.name_1.y vs EETests.FORM.name_2.x': 0.0,
                                     'EETests.FORM.name_2.y vs EETests.FORM.name_2.x': 0.3000000000010772,
                                     'EETests.FORM.name_1.z vs EETests.FORM.name_2.x': 0.3000000000010772,
                                     'EETests.FORM.name_2.z vs EETests.FORM.name_2.x': 0.3000000000010772,
                                     'EETests.FORM.name_1.y vs EETests.FORM.Disc1.name_1.b': 0.4999999999988347,
                                     'EETests.FORM.name_2.y vs EETests.FORM.Disc1.name_1.b': 0.0,
                                     'EETests.FORM.name_1.z vs EETests.FORM.Disc1.name_1.b': 0.0,
                                     'EETests.FORM.name_2.z vs EETests.FORM.Disc1.name_1.b': 0.0,
                                     'EETests.FORM.name_1.y vs EETests.FORM.Disc1.name_2.b': 0.0,
                                     'EETests.FORM.name_2.y vs EETests.FORM.Disc1.name_2.b': 0.4999999999988347,
                                     'EETests.FORM.name_1.z vs EETests.FORM.Disc1.name_2.b': 0.4999999999988347,
                                     'EETests.FORM.name_2.z vs EETests.FORM.Disc1.name_2.b': 0.4999999999988347,
                                     'EETests.FORM.name_1.y vs EETests.FORM.Disc2.name_1.constant': -0.0,
                                     'EETests.FORM.name_2.y vs EETests.FORM.Disc2.name_1.constant': -0.0,
                                     'EETests.FORM.name_1.z vs EETests.FORM.Disc2.name_1.constant': -0.0,
                                     'EETests.FORM.name_2.z vs EETests.FORM.Disc2.name_1.constant': -0.0,
                                     'EETests.FORM.name_1.y vs EETests.FORM.Disc2.name_2.constant': -0.0,
                                     'EETests.FORM.name_2.y vs EETests.FORM.Disc2.name_2.constant': -0.0,
                                     'EETests.FORM.name_1.z vs EETests.FORM.Disc2.name_2.constant': -0.0,
                                     'EETests.FORM.name_2.z vs EETests.FORM.Disc2.name_2.constant': -0.0}

        self.assertDictEqual(form_output_ref_5_percent, form_output['5.0%'],
                             'The comparison with the reference is not correct')
        # Check on graphs :
        disc = self.exec_eng.dm.get_disciplines_with_name(
            'EETests.FORM')[0]
        filters = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filters)
#         for graph in graph_list:
#             graph.to_plotly().show()

    def test_09_sensitivity_analysis_with_all_types(self):
        '''
            Compare a sensitivity analysis with the same execution
        '''
        self.sub_proc = 'test_discall_types'
        builder_list = self.exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                      mod_id=self.sub_proc)

        sa_builder = self.exec_eng.factory.create_evaluator_builder(
            'SA', 'sensitivity', builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(sa_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        values_dict = {}
        values_dict['EETests.SA.eval_inputs'] = ['z']
        values_dict['EETests.SA.eval_outputs'] = [
            'df_out', 'dict_out', 'dict_df_out', 'dict_dict_out']
        values_dict['EETests.SA.variation_list'] = ['+/-5%']
        values_dict['EETests.z'] = 1.0
        values_dict['EETests.y'] = 2.0
        values_dict['EETests.AC_list'] = ['A1', 'A2', 'EETests']

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

        sensitivity_output = self.exec_eng.dm.get_value(
            'EETests.SA.sensitivity_outputs')

        # Check on graphs :
        disc = self.exec_eng.dm.get_disciplines_with_name(
            'EETests.SA')[0]
        filters = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filters)
#         for graph in graph_list:
#             graph.to_plotly().show()

        values_dict['EETests.SA.eval_inputs'] = ['y', 'z']

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

        # Check on graphs :
        disc = self.exec_eng.dm.get_disciplines_with_name(
            'EETests.SA')[0]
        filters = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filters)
#         for graph in graph_list:
#             graph.to_plotly().show()

    def test_10_FORM_analysis_with_all_types(self):
        '''
            Compare a sensitivity analysis with the same execution
        '''
        self.sub_proc = 'test_discall_types'
        builder_list = self.exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                      mod_id=self.sub_proc)

        sa_builder = self.exec_eng.factory.create_evaluator_builder(
            'FORM', 'FORM', builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(sa_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        values_dict = {}
        values_dict['EETests.FORM.eval_inputs'] = ['z']
        values_dict['EETests.FORM.eval_outputs'] = [
            'df_out', 'dict_out', 'dict_df_out', 'dict_dict_out']
        values_dict['EETests.FORM.variation_list'] = ['+/-5%']
        values_dict['EETests.FORM.grad_method'] = '2nd order FD'
        values_dict['EETests.z'] = 1.0
        values_dict['EETests.y'] = 2.0
        values_dict['EETests.AC_list'] = ['A1', 'A2', 'EETests']

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

        sensitivity_output = self.exec_eng.dm.get_value(
            'EETests.FORM.FORM_outputs')

        # Check on graphs :
        disc = self.exec_eng.dm.get_disciplines_with_name(
            'EETests.FORM')[0]
        filters = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filters)
#         for graph in graph_list:
#             graph.to_plotly().show()

        values_dict['EETests.FORM.eval_inputs'] = ['y', 'z']

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

        # Check on graphs :
        disc = self.exec_eng.dm.get_disciplines_with_name(
            'EETests.FORM')[0]
        filters = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filters)
#         for graph in graph_list:
#             graph.to_plotly().show()

    def test_11_grad_analysis_with_all_types(self):
        '''
            Compare a sensitivity analysis with the same execution
        '''
        self.sub_proc = 'test_discall_types'
        builder_list = self.exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                      mod_id=self.sub_proc)

        sa_builder = self.exec_eng.factory.create_evaluator_builder(
            'gradient', 'gradient', builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(sa_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        values_dict = {}
        values_dict['EETests.gradient.eval_inputs'] = ['z']
        values_dict['EETests.gradient.eval_outputs'] = [
            'df_out', 'dict_out', 'dict_df_out', 'dict_dict_out']
        values_dict['EETests.gradient.grad_method'] = '2nd order FD'
        values_dict['EETests.z'] = 1.0
        values_dict['EETests.y'] = 2.0
        values_dict['EETests.AC_list'] = ['A1', 'A2']
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

        sensitivity_output = self.exec_eng.dm.get_value(
            'EETests.gradient.gradient_outputs')

        # Check on graphs :
        disc = self.exec_eng.dm.get_disciplines_with_name(
            'EETests.gradient')[0]
        filters = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filters)
#         for graph in graph_list:
#             graph.to_plotly().show()

        values_dict['EETests.gradient.eval_inputs'] = ['y', 'z']
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

        # Check on graphs :
        disc = self.exec_eng.dm.get_disciplines_with_name(
            'EETests.gradient')[0]
        filters = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filters)
#         for graph in graph_list:
#             graph.to_plotly().show()
