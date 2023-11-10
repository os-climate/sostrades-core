'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/10-2023/11/03 Copyright 2023 Capgemini

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

from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """
        Usecase for lhs DoE and Eval on x variable of Sellar Problem
        """

        disc1_name = 'Disc1'
        ns = f'{self.study_name}'
        dspace_dict = {'variable': [f'subprocess.{disc1_name}.a', 'x'],

                       'lower_bnd': [0., 0.],
                       'upper_bnd': [10., 10.],

                       }
        dspace = pd.DataFrame(dspace_dict)

        # input_selection_x_z = {'selected_input': [True, False, True, False],
        #                        'full_name': [f'subprocess.{disc1_name}.a', f'subprocess.{disc1_name}.b',
        #                                      f'x',
        #                                      f'subprocess.Disc2.power']}
        # input_selection_x_z = pd.DataFrame(input_selection_x_z)

        output_selection_obj_y1_y2 = {'selected_output': [True, True, False],
                                      'full_name': [f'subprocess.{disc1_name}.indicator', 'z', 'y']}
        output_selection_obj_y1_y2 = pd.DataFrame(output_selection_obj_y1_y2)

        disc_dict = {}
        # DoE inputs
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = 'cartesian_product'

        a_list = np.linspace(0, 10, 10).tolist()
        x_list = np.linspace(0, 10, 10).tolist()
        eval_inputs = pd.DataFrame({'selected_input': [True, False, True, False],
                                    'full_name': [f'subprocess.{disc1_name}.a', f'subprocess.{disc1_name}.b',
                                                  f'x', f'subprocess.Disc2.power']})

        eval_inputs_cp = pd.DataFrame({'selected_input': [True, False, True, False],
                                       'full_name': [f'subprocess.{disc1_name}.a', f'subprocess.{disc1_name}.b',
                                                     f'x', f'subprocess.Disc2.power'],
                                       'list_of_values': [a_list, [], x_list, []]
                                       })
        disc_dict[f'{self.study_name}.Eval.with_sample_generator'] = True
        disc_dict[f'{self.study_name}.Eval.eval_inputs'] = eval_inputs_cp
        disc_dict[f'{self.study_name}.Eval.eval_inputs'] = eval_inputs
        disc_dict[f'{ns}.Eval.design_space'] = dspace
        # disc_dict[f'{ns}.Eval.eval_inputs'] = input_selection_x_z
        disc_dict[f'{ns}.Eval.eval_outputs'] = output_selection_obj_y1_y2

        disc_dict[f'{ns}.Eval.x'] = 10.
        disc_dict[f'{ns}.Eval.subprocess.{disc1_name}.a'] = 5.
        disc_dict[f'{ns}.Eval.subprocess.{disc1_name}.b'] = 2.
        disc_dict[f'{ns}.Eval.subprocess.Disc2.constant'] = 3.1416
        disc_dict[f'{ns}.Eval.subprocess.Disc2.power'] = 2

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes()
    uc_cls.run()
    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.proxy_disciplines:
        if disc.sos_name.endswith('UncertaintyQuantification'):
            filters = ppf.get_post_processing_filters_by_discipline(
                disc)
            graph_list = ppf.get_post_processing_by_discipline(
                disc, filters, as_json=False)

            for graph in graph_list:
                graph.to_plotly().show()
