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

from os.path import join, dirname
from copy import deepcopy

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
from sostrades_core.sos_processes.test.test_sellar_sub_opt_w_design_var.usecase import Study
from sostrades_core.sos_processes.test.test_sellar_opt_w_design_var_sub.usecase import Study as study_sellar_sub


class SellarOptimSubprocessJacobianDiscTest(AbstractJacobianUnittest):
    """
    Class to test Sellar sub process derivatives in several cases.
    - test_01_gradient_subprocess_double_level_coupling : coupling inside another coupling process
    - test_02_gradient_subprocess_flatten_input_data : coupling of sellar disciplines and Design Var + Func Manager flattened
                    with input data as input of check_jacobian
    - test_03_gradient_subprocess_flatten_local_data : use local data as input of check jacobian
    - test_04_gradient_subprocess_flatten_input_data_cache : use input data and cache
    - test_05_gradient_subprocess_flatten_input_data_cache_and_warmstart : use input data and cache and warmstart
    - test_06_gradient_subprocess_flatten_local_data_deepcopy : deepcopy local data before check jacobian
    - test_07_gradient_subprocess_flatten_local_data_different_exec_engine : execute but use another exec engine for check jacobain

    """

    def analytic_grad_entry(self):
        return [self._test_01_gradient_subprocess_double_level_coupling(),
                ]

    def setUp(self):
        self.name = 'Test'

    def _test_01_gradient_subprocess_double_level_coupling(self):
        """
        Test objective lagrangian derivative using a double level coupling of sellar process.
        """
        # create new exec engine
        ee = ExecutionEngine(self.name)
        # get builder from process
        builder = ee.factory.get_builder_from_process('sostrades_core.sos_processes.test',
                                                      'test_sellar_sub_opt_w_design_var')
        ee.factory.set_builders_to_coupling_builder(builder)
        # configure
        ee.configure()
        # import study and set data
        usecase = Study(execution_engine=ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        ee.load_study_from_input_dict(full_values_dict)
        # call the two methods used before an execute (to ensure that the execution in the linearize step is done in
        # the same conditions as a normal execution)
        ee.update_from_dm()
        ee.prepare_execution()
        ee.display_treeview_nodes()
        coupling_disc = ee.root_process.proxy_disciplines[0]

        # check derivative of objective lagrangian wrt x_in and z_in
        pkl_name = f'jacobian_obj_vs_design_var_sellar_test_01.pkl'
        inputs = ['Test.Sellar.SellarOptimScenario.x_in', 'Test.Sellar.SellarOptimScenario.z_in']
        outputs = ['Test.Sellar.SellarOptimScenario.objective_lagrangian']
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True
        self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                            discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                            step=1.0e-4, derr_approx='finite_differences', threshold=1e-15,
                            local_data=full_values_dict,
                            # coupling_disc.mdo_discipline_wrapp.mdo_discipline.local_data,
                            inputs=inputs,
                            outputs=outputs)

    def test_02_gradient_subprocess_flatten_input_data(self):
        """
        Test objective lagrangian derivative using a one level coupling of sellar process.
        """
        # create new exec engine
        ee = ExecutionEngine(self.name)
        # get builder from process
        builder = ee.factory.get_builder_from_process('sostrades_core.sos_processes.test',
                                                      'test_sellar_opt_w_design_var_sub')
        ee.factory.set_builders_to_coupling_builder(builder)
        ee.configure()
        # import study and set data
        usecase = study_sellar_sub(execution_engine=ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        # set cache to None and warm start to False
        full_values_dict['Test.SellarCoupling.warm_start'] = False
        full_values_dict['Test.SellarCoupling.cache_type'] = None  # 'SimpleCache'
        full_values_dict['Test.SellarCoupling.propagate_cache_to_children'] = True
        full_values_dict['Test.warm_start'] = False
        full_values_dict['Test.cache_type'] = None  # 'SimpleCache'
        full_values_dict['Test.propagate_cache_to_children'] = True
        full_values_dict['Test.SellarCoupling.max_mda_iter'] = 30
        full_values_dict['Test.SellarCoupling.tolerance'] = 1e-14
        full_values_dict['Test.SellarCoupling.sub_mda_class'] = 'MDAGaussSeidel'
        # load_data
        ee.load_study_from_input_dict(full_values_dict)
        # call the two methods used before an execute (to ensure that the execution in the linearize step is done in
        # the same conditions as a normal execution)
        ee.update_from_dm()
        ee.prepare_execution()
        ee.display_treeview_nodes()
        coupling_disc = ee.root_process.proxy_disciplines[0]

        pkl_name = f'jacobian_obj_vs_design_var_sellar_test_02.pkl'
        inputs = ['Test.x_in', 'Test.z_in']
        outputs = ['Test.objective_lagrangian']
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True

        self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                            discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                            step=1.0e-14, derr_approx='complex_step', threshold=1e-10,
                            local_data=full_values_dict,
                            inputs=inputs,
                            outputs=outputs)

    def _test_03_gradient_subprocess_flatten_local_data(self):
        """
        Test objective lagrangian derivative using a one level coupling of sellar process and local data.
        """
        # create new exec engine
        ee = ExecutionEngine(self.name)
        # get builder from process
        builder = ee.factory.get_builder_from_process('sostrades_core.sos_processes.test',
                                                      'test_sellar_opt_w_design_var_sub')
        ee.factory.set_builders_to_coupling_builder(builder)
        ee.configure()
        # import study and set data
        usecase = study_sellar_sub(execution_engine=ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        # set cache to None and warm start to False
        full_values_dict['Test.SellarCoupling.warm_start'] = False
        full_values_dict['Test.SellarCoupling.cache_type'] = None  # 'SimpleCache'
        full_values_dict['Test.SellarCoupling.propagate_cache_to_children'] = True
        full_values_dict['Test.warm_start'] = False
        full_values_dict['Test.cache_type'] = None  # 'SimpleCache'
        full_values_dict['Test.propagate_cache_to_children'] = True
        full_values_dict['Test.SellarCoupling.max_mda_iter'] = 30
        full_values_dict['Test.SellarCoupling.tolerance'] = 1e-14
        full_values_dict['Test.SellarCoupling.sub_mda_class'] = 'MDAGaussSeidel'
        # load_data
        ee.load_study_from_input_dict(full_values_dict)
        # execute
        ee.execute()
        ee.display_treeview_nodes()
        coupling_disc = ee.root_process.proxy_disciplines[0]

        pkl_name = f'jacobian_obj_vs_design_var_sellar_test_03.pkl'
        inputs = ['Test.x_in', 'Test.z_in']
        outputs = ['Test.objective_lagrangian']
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True

        self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                            discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                            step=1.0e-14, derr_approx='complex_step', threshold=1e-10,
                            local_data=coupling_disc.mdo_discipline_wrapp.mdo_discipline.local_data,
                            inputs=inputs,
                            outputs=outputs)

    def test_04_gradient_subprocess_flatten_input_data_cache(self):
        """
        Test objective lagrangian derivative using a one level coupling of sellar process and input data with cache.
        """
        # create new exec engine
        ee = ExecutionEngine(self.name)
        # get builder from process
        builder = ee.factory.get_builder_from_process('sostrades_core.sos_processes.test',
                                                      'test_sellar_opt_w_design_var_sub')
        ee.factory.set_builders_to_coupling_builder(builder)
        ee.configure()
        # import study and set data
        usecase = study_sellar_sub(execution_engine=ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        # set cache to None and warm start to False
        full_values_dict['Test.SellarCoupling.warm_start'] = False
        full_values_dict['Test.SellarCoupling.cache_type'] = 'SimpleCache'
        full_values_dict['Test.SellarCoupling.propagate_cache_to_children'] = True
        full_values_dict['Test.warm_start'] = False
        full_values_dict['Test.cache_type'] = 'SimpleCache'
        full_values_dict['Test.propagate_cache_to_children'] = True
        full_values_dict['Test.SellarCoupling.max_mda_iter'] = 30
        full_values_dict['Test.SellarCoupling.tolerance'] = 1e-14
        full_values_dict['Test.SellarCoupling.sub_mda_class'] = 'MDAGaussSeidel'
        # load_data
        ee.load_study_from_input_dict(full_values_dict)
        # call the two methods used before an execute (to ensure that the execution in the linearize step is done in
        # the same conditions as a normal execution)
        ee.update_from_dm()
        ee.prepare_execution()
        ee.display_treeview_nodes()
        coupling_disc = ee.root_process.proxy_disciplines[0]

        pkl_name = f'jacobian_obj_vs_design_var_sellar_test_04.pkl'
        inputs = ['Test.x_in', 'Test.z_in']
        outputs = ['Test.objective_lagrangian']
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True

        self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                            discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                            step=1.0e-14, derr_approx='complex_step', threshold=1e-10,
                            local_data=full_values_dict,
                            inputs=inputs,
                            outputs=outputs)

    def _test_05_gradient_subprocess_flatten_input_data_cache_and_warmstart(self):
        """
        Test objective lagrangian derivative using a one level coupling of sellar process and input data with
        cache and warmstart.
        """
        # create new exec engine
        ee = ExecutionEngine(self.name)
        # get builder from process
        builder = ee.factory.get_builder_from_process('sostrades_core.sos_processes.test',
                                                      'test_sellar_opt_w_design_var_sub')
        ee.factory.set_builders_to_coupling_builder(builder)
        ee.configure()
        # import study and set data
        usecase = study_sellar_sub(execution_engine=ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        # set cache to None and warm start to False
        full_values_dict['Test.SellarCoupling.warm_start'] = True
        full_values_dict['Test.SellarCoupling.cache_type'] = 'SimpleCache'
        full_values_dict['Test.SellarCoupling.propagate_cache_to_children'] = True
        full_values_dict['Test.warm_start'] = True
        full_values_dict['Test.cache_type'] = 'SimpleCache'
        full_values_dict['Test.propagate_cache_to_children'] = True
        full_values_dict['Test.SellarCoupling.max_mda_iter'] = 30
        full_values_dict['Test.SellarCoupling.tolerance'] = 1e-14
        full_values_dict['Test.SellarCoupling.sub_mda_class'] = 'MDAGaussSeidel'
        # load_data
        ee.load_study_from_input_dict(full_values_dict)
        # call the two methods used before an execute (to ensure that the execution in the linearize step is done in
        # the same conditions as a normal execution)
        ee.update_from_dm()
        ee.prepare_execution()
        ee.display_treeview_nodes()
        coupling_disc = ee.root_process.proxy_disciplines[0]

        pkl_name = f'jacobian_obj_vs_design_var_sellar_test_05.pkl'
        inputs = ['Test.x_in', 'Test.z_in']
        outputs = ['Test.objective_lagrangian']
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True

        self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                            discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                            step=1.0e-14, derr_approx='complex_step', threshold=1e-10,
                            local_data=full_values_dict,
                            inputs=inputs,
                            outputs=outputs)

    def _test_06_gradient_subprocess_flatten_local_data_deepcopy(self):
        """
        Test objective lagrangian derivative using a one level coupling of sellar process and local data with deepcopy.
        """
        # create new exec engine
        ee = ExecutionEngine(self.name)
        # get builder from process
        builder = ee.factory.get_builder_from_process('sostrades_core.sos_processes.test',
                                                      'test_sellar_opt_w_design_var_sub')
        ee.factory.set_builders_to_coupling_builder(builder)
        ee.configure()
        # import study and set data
        usecase = study_sellar_sub(execution_engine=ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        # set cache to None and warm start to False
        full_values_dict['Test.SellarCoupling.warm_start'] = False
        full_values_dict['Test.SellarCoupling.cache_type'] = None  # 'SimpleCache'
        full_values_dict['Test.SellarCoupling.propagate_cache_to_children'] = True
        full_values_dict['Test.warm_start'] = False
        full_values_dict['Test.cache_type'] = None  # 'SimpleCache'
        full_values_dict['Test.propagate_cache_to_children'] = True
        full_values_dict['Test.SellarCoupling.max_mda_iter'] = 30
        full_values_dict['Test.SellarCoupling.tolerance'] = 1e-14
        full_values_dict['Test.SellarCoupling.sub_mda_class'] = 'MDAGaussSeidel'
        # load_data
        ee.load_study_from_input_dict(full_values_dict)
        # execute
        ee.execute()
        ee.display_treeview_nodes()
        coupling_disc = ee.root_process.proxy_disciplines[0]

        pkl_name = f'jacobian_obj_vs_design_var_sellar_test_06.pkl'
        inputs = ['Test.x_in', 'Test.z_in']
        outputs = ['Test.objective_lagrangian']
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True
        local_data_after_execute = coupling_disc.mdo_discipline_wrapp.mdo_discipline.local_data
        self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                            discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                            step=1.0e-14, derr_approx='complex_step', threshold=1e-10,
                            local_data= deepcopy(local_data_after_execute),
                            inputs=inputs,
                            outputs=outputs)

    def _test_07_gradient_subprocess_flatten_local_data_different_exec_engine(self):
        """
        Use local data on sellar sub process that comes from the execution of Sellar from another test
        """
        # create new exec engine
        ee = ExecutionEngine(self.name)
        # get builder from process
        builder = ee.factory.get_builder_from_process('sostrades_core.sos_processes.test',
                                                      'test_sellar_opt_w_design_var_sub')
        ee.factory.set_builders_to_coupling_builder(builder)
        ee.configure()
        # import study and set data
        usecase = study_sellar_sub(execution_engine=ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        # set cache to None and warm start to False
        full_values_dict['Test.SellarCoupling.warm_start'] = False
        full_values_dict['Test.SellarCoupling.cache_type'] = None  # 'SimpleCache'
        full_values_dict['Test.SellarCoupling.propagate_cache_to_children'] = True
        full_values_dict['Test.warm_start'] = False
        full_values_dict['Test.cache_type'] = None  # 'SimpleCache'
        full_values_dict['Test.propagate_cache_to_children'] = True
        full_values_dict['Test.SellarCoupling.max_mda_iter'] = 30
        full_values_dict['Test.SellarCoupling.tolerance'] = 1e-14
        full_values_dict['Test.SellarCoupling.sub_mda_class'] = 'MDAGaussSeidel'
        # load_data
        ee.load_study_from_input_dict(full_values_dict)
        # call the two methods used before an execute (to ensure that the execution in the linearize step is done in
        # the same conditions as a normal execution)
        ee.update_from_dm()
        ee.prepare_execution()
        ee.display_treeview_nodes()
        coupling_disc = ee.root_process.proxy_disciplines[0]

        pkl_name = f'jacobian_obj_vs_design_var_sellar_test_07.pkl'
        inputs = ['Test.x_in', 'Test.z_in']
        outputs = ['Test.objective_lagrangian']
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True
        local_data = self.execute_sellar()
        self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                            discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                            step=1.0e-14, derr_approx='complex_step', threshold=1e-10,
                            local_data=local_data,
                            inputs=inputs,
                            outputs=outputs)

    def execute_sellar(self):
        """
        Function to execute sellar with a newly created exec engine
        Called only in test_07
        """
        # create new exec engine and execute sellar sub process
        ee = ExecutionEngine(self.name)
        builder = ee.factory.get_builder_from_process('sostrades_core.sos_processes.test',
                                                      'test_sellar_opt_w_design_var_sub')
        ee.factory.set_builders_to_coupling_builder(builder)
        ee.configure()
        usecase = study_sellar_sub(execution_engine=ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        full_values_dict['Test.SellarCoupling.warm_start'] = False
        full_values_dict['Test.SellarCoupling.cache_type'] = None  # 'SimpleCache'
        full_values_dict['Test.SellarCoupling.propagate_cache_to_children'] = True
        full_values_dict['Test.warm_start'] = False
        full_values_dict['Test.cache_type'] = None  # 'SimpleCache'
        full_values_dict['Test.propagate_cache_to_children'] = True
        full_values_dict['Test.SellarCoupling.max_mda_iter'] = 30
        full_values_dict['Test.SellarCoupling.tolerance'] = 1e-14
        full_values_dict['Test.SellarCoupling.sub_mda_class'] = 'MDAGaussSeidel'

        ee.load_study_from_input_dict(full_values_dict)
        ee.execute()
        ee.display_treeview_nodes()
        coupling_disc = ee.root_process.proxy_disciplines[0]

        return deepcopy(coupling_disc.mdo_discipline_wrapp.mdo_discipline.local_data)