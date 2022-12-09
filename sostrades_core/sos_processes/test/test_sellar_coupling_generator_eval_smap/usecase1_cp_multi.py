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
import pandas as pd
from numpy import array

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """
        Usecase for lhs DoE and Eval on x variable of Sellar Problem
        """

        ns = f'{self.study_name}'

        dict_of_list_values = {
            'SellarCoupling.x': [array([3.]), array([4.])],
            'SellarCoupling.z': [array([-10., 0.])],
            'SellarCoupling.Sellar_Problem.local_dv': [10.],
            'SellarCoupling.y_1': [array([1.])],
            'SellarCoupling.y_2': [array([1.])]
        }
        list_of_values = [dict_of_list_values['SellarCoupling.Sellar_Problem.local_dv'], dict_of_list_values['SellarCoupling.x'],
                          dict_of_list_values['SellarCoupling.y_1'], dict_of_list_values['SellarCoupling.y_2'], dict_of_list_values['SellarCoupling.z']]

        input_selection_cp_x_z = {'selected_input': [True, True, True, True, True],
                                  'full_name': ['SellarCoupling.Sellar_Problem.local_dv', 'SellarCoupling.x', 'SellarCoupling.y_1',
                                                'SellarCoupling.y_2',
                                                'SellarCoupling.z'],
                                  'list_of_values': list_of_values
                                  }
        input_selection_cp_x_z = pd.DataFrame(input_selection_cp_x_z)

        repo = 'sostrades_core.sos_processes.test'
        mod_id = 'test_sellar_coupling'
        my_usecase = 'usecase'
        anonymize_input_dict_from_usecase = self.static_load_raw_usecase_data(
            repo, mod_id, my_usecase)

        disc_dict = {}
        # CP + Eval inputs
        disc_dict[f'{ns}.Eval.builder_mode'] = 'multi_instance'
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = 'cartesian_product'
        disc_dict[f'{ns}.Eval.eval_inputs_cp'] = input_selection_cp_x_z
        disc_dict[f'{ns}.Eval.usecase_data'] = anonymize_input_dict_from_usecase
        disc_dict[f'{ns}.Eval.instance_reference'] = True

        # Sellar referene inputs
        local_dv = 10.
        disc_dict[f'{ns}.Eval.ReferenceScenario.SellarCoupling.x'] = array([
                                                                           2.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.SellarCoupling.y_1'] = array([
                                                                             1.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.SellarCoupling.y_2'] = array([
                                                                             1.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.SellarCoupling.z'] = array([
                                                                           1., 1.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.SellarCoupling.Sellar_Problem.local_dv'] = local_dv

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)

    my_disc = uc_cls.execution_engine.dm.get_disciplines_with_name(
        'usecase1_cp_multi.Eval.ReferenceScenario.SellarCoupling.Sellar_1')[0]
    a1 = my_disc.get_data_io_from_key('in', 'x')['value']
    a2 = my_disc.get_data_io_from_key('in', 'y_2')['value']
    a3 = my_disc.get_data_io_from_key('in', 'z')['value']

    my_disc = uc_cls.execution_engine.dm.get_disciplines_with_name(
        'usecase1_cp_multi.Eval.ReferenceScenario.SellarCoupling.Sellar_2')[0]
    a4 = my_disc.get_data_io_from_key('in', 'y_1')['value']
    a5 = my_disc.get_data_io_from_key('in', 'z')['value']

    my_disc = uc_cls.execution_engine.dm.get_disciplines_with_name(
        'usecase1_cp_multi.Eval.ReferenceScenario.SellarCoupling.Sellar_Problem')[0]
    a6 = my_disc.get_data_io_from_key('in', 'x')['value']
    a7 = my_disc.get_data_io_from_key('in', 'y_1')['value']
    a8 = my_disc.get_data_io_from_key('in', 'y_2')['value']
    a9 = my_disc.get_data_io_from_key('in', 'z')['value']
    a10 = my_disc.get_data_io_from_key('in', 'local_dv')['value']

    a11 = uc_cls.ee.dm.get_value(
        'usecase1_cp_multi.Eval.ReferenceScenario.SellarCoupling.x')
    a12 = uc_cls.ee.dm.get_value(
        'usecase1_cp_multi.Eval.ReferenceScenario.SellarCoupling.z')
    a13 = uc_cls.ee.dm.get_value(
        'usecase1_cp_multi.Eval.ReferenceScenario.SellarCoupling.y_1')
    a14 = uc_cls.ee.dm.get_value(
        'usecase1_cp_multi.Eval.ReferenceScenario.SellarCoupling.y_2')
    a15 = uc_cls.ee.dm.get_value(
        'usecase1_cp_multi.Eval.scenario_1.SellarCoupling.x')
    a16 = uc_cls.ee.dm.get_value(
        'usecase1_cp_multi.Eval.scenario_1.SellarCoupling.z')
    a17 = uc_cls.ee.dm.get_value(
        'usecase1_cp_multi.Eval.generated_samples')
    # print(a17.to_markdown())
    a18 = uc_cls.ee.dm.get_value(
        'usecase1_cp_multi.Eval.scenario_df')
    # print(a18.to_markdown())
    a19 = uc_cls.ee.dm.get_value(
        'usecase1_cp_multi.Eval.usecase_data')
    import pprint
    pprint.pprint(a19)

    uc_cls.run()
