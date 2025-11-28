'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/10-2025/11/28 Copyright 2025 Capgemini

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
from sostrades_core.tools.proc_builder.process_builder_parameter_type import (
    ProcessBuilderParameterType,
)


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None) -> None:
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """Usecase for lhs DoE and Eval on x variable of a Problem"""
        coupling_name = 'CostCoupling'

        ns = f'{self.study_name}'

        dspace_dict = {'variable': [f'{coupling_name}.cost_problem.weight_factor'],
                       'lower_bnd': [0.8],
                       'upper_bnd': [2.0],
                       }


        dspace = pd.DataFrame(dspace_dict)

        input_selection_x = {'selected_input': [True, False, False, False, False],
                             'full_name': [f'{coupling_name}.cost_problem.weight_factor', f'{coupling_name}.engine_power', f'{coupling_name}.manufacturing_cost',
                                           f'{coupling_name}.maintenance_cost',
                                           f'{coupling_name}.material_specs']}

        input_selection_x = pd.DataFrame(input_selection_x)

        output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
                                      'full_name': [f'{coupling_name}.quality_constraint', f'{coupling_name}.budget_constraint', f'{coupling_name}.total_cost',
                                                    f'{coupling_name}.manufacturing_cost', f'{coupling_name}.maintenance_cost']}

        output_selection_obj_y1_y2 = pd.DataFrame(output_selection_obj_y1_y2)

        disc_dict = {}
        # DoE + Eval inputs
        n_samples = 20
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = 'doe_algo'
        disc_dict[f'{ns}.SampleGenerator.sampling_generation_mode'] = 'at_run_time'
        disc_dict[f'{ns}.SampleGenerator.sampling_algo'] = "PYDOE_LHS"
        disc_dict[f'{ns}.SampleGenerator.design_space'] = dspace
        disc_dict[f'{ns}.SampleGenerator.algo_options'] = {
            'n_samples': n_samples}
        disc_dict[f'{ns}.Eval.with_sample_generator'] = True
        disc_dict[f'{ns}.SampleGenerator.eval_inputs'] = input_selection_x
        disc_dict[f'{ns}.Eval.gather_outputs'] = output_selection_obj_y1_y2
        disc_dict[f'{ns}.Eval.n_processes'] = 1

        with_modal = True
        anonymize_input_dict_from_usecase = {}
        if with_modal:
            repo = 'sostrades_core.sos_processes.test'
            mod_id = 'test_problem_coupling_dataframes'
            my_usecase = 'Empty'
            process_builder_parameter_type = ProcessBuilderParameterType(
                mod_id, repo, my_usecase)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            disc_dict[f'{ns}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            disc_dict[f'{ns}.Eval.usecase_data'] = anonymize_input_dict_from_usecase

        # Car cost computation inputs - same values as other usecase
        weight_factor = 1.2

        # Initialize dataframes for car cost computation over years 2025-2030
        manufacturing_cost_df = pd.DataFrame({
            'years': np.arange(2025, 2031),
            'value': [8000.0, 8200.0, 8400.0, 8600.0, 8800.0, 9000.0]  # Increasing manufacturing costs
        })

        maintenance_cost_df = pd.DataFrame({
            'years': np.arange(2025, 2031),
            'value': [2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0]  # Increasing maintenance costs over time
        })

        # Engine power specifications (different engine sizes for DOE)
        engine_power_dict = {'years': np.arange(1, 5), 'value': np.array([150.0, 200.0, 250.0, 300.0])}  # HP values

        disc_dict[f'{ns}.Eval.{coupling_name}.engine_power'] = engine_power_dict
        disc_dict[f'{ns}.Eval.{coupling_name}.manufacturing_cost'] = manufacturing_cost_df
        disc_dict[f'{ns}.Eval.{coupling_name}.maintenance_cost'] = maintenance_cost_df
        disc_dict[f'{ns}.Eval.{coupling_name}.material_specs'] = np.array([3.0, 2.5])  # [quality_factor, durability_factor]
        disc_dict[f'{ns}.Eval.{coupling_name}.cost_problem.weight_factor'] = weight_factor

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
