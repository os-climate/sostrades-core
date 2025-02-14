'''
Copyright 2025 Capgemini

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
import numpy as np
import pandas as pd

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):
    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """Usecase for lhs DoE and Eval on x variable of Sellar Problem"""
        ns = self.study_name
        coupling_name = 'SellarCoupling'

        input_selection_a = {
            "selected_input": [True],
            "full_name": ["SellarCoupling.Sellar_Problem.local_dv"],
        }
        input_selection_a = pd.DataFrame(input_selection_a)

        output_selection_ind = {
            "selected_output": [True],
            "full_name": ["SellarCoupling.obj"],
        }
        output_selection_ind = pd.DataFrame(output_selection_ind)

        disc_dict = {}
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = "tornado_chart_analysis"
        disc_dict[f'{ns}.SampleGenerator.variation_list'] = [-10.0, 10.0]
        disc_dict[f'{ns}.Eval.with_sample_generator'] = True
        disc_dict[f'{ns}.SampleGenerator.eval_inputs'] = input_selection_a
        disc_dict[f'{ns}.Eval.gather_outputs'] = output_selection_ind
        disc_dict[f'{ns}.Eval.{coupling_name}.x'] = np.array([1.])
        disc_dict[f'{ns}.Eval.{coupling_name}.y_2'] = np.array([1.])
        disc_dict[f'{ns}.Eval.{coupling_name}.z'] = np.array([1., 1.])
        disc_dict[f'{ns}.Eval.{coupling_name}.Sellar_Problem.local_dv'] = 10.
        disc_dict[f'{ns}.Eval.{coupling_name}.max_mda_iter'] = 100
        disc_dict[f'{ns}.Eval.{coupling_name}.tolerance'] = 1e-12
        return [disc_dict]


if "__main__" == __name__:
    ns = "usecase_tornado_analysis"
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
