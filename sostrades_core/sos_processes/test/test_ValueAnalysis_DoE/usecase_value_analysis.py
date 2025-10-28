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

import pandas as pd

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, run_usecase=True, execution_engine=None) -> None:
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """Setup DoE configuration for FakeCarModel with ValueAnalysis"""
        ns = self.study_name
        # Design space ranges for automotive parameters
        # =============================================
        # SIMPLEDOE + VALUEANALYSIS CONFIGURATION
        # =============================================
        # Using our working DoE approach that bypasses DoE.Eval issues
        #['FakeCarModel.aerodynamic_coefficient', 'FakeCarModel.battery_capacity', 'FakeCarModel.brake_type',
        # 'FakeCarModel.engine_power', 'FakeCarModel.engine_type', 'FakeCarModel.fuel_tank_capacity',
        # 'FakeCarModel.tire_type', 'FakeCarModel.transmission_type', 'FakeCarModel.vehicle_weight',
        # 'FakeCarModel.wheel_diameter']
        input_list =['Eval.FakeCarModel.battery_capacity','FakeCarModel.vehicle_weight','FakeCarModel.aerodynamic_coefficient',
                                    'FakeCarModel.engine_power','FakeCarModel.wheel_diameter']
        dspace_dict = {'variable':input_list ,

                       'lower_bnd': [30.,1200.,0.2,100.0,16.0],
                       'upper_bnd': [100.,2500.,0.45,350.0,20.0],

                       }
        dspace = pd.DataFrame(dspace_dict)

        gather_outputs = {'selected_output': [True, True,True],
                                'full_name': ['FakeCarModel.top_speed', 'FakeCarModel.range','FakeCarModel.manufacturing_cost']}
        gather_outputs = pd.DataFrame(gather_outputs)
        eval_inputs = {'selected_input': [True]*len(input_list),
                             'full_name': input_list}
        eval_inputs = pd.DataFrame(eval_inputs)
        config_dict = {
            # DoE Configuration - parameters that define the DoE space
            f'{ns}.Eval.with_sample_generator': True,
            f'{ns}.SampleGenerator.sampling_method': 'doe_algo',
        f'{ns}.SampleGenerator.sampling_algo':  "PYDOE_LHS",
            f'{ns}.SampleGenerator.design_space': dspace,
        f'{ns}.SampleGenerator.algo_options':{
            'n_samples': 8},
            f'{ns}.Eval.gather_outputs': gather_outputs,
            f'{ns}.SampleGenerator.eval_inputs': eval_inputs,
            # Fixed FakeCarModel parameters for all DoE samples
            f'{ns}.Eval.FakeCarModel.engine_type': 'Electric',
            f'{ns}.Eval.FakeCarModel.brake_type': 'Disc',
            f'{ns}.Eval.FakeCarModel.tire_type': 'Performance',
            f'{ns}.Eval.FakeCarModel.transmission_type': 'Automatic',
            f'{ns}.Eval.FakeCarModel.fuel_tank_capacity': 0.0,  # Electric vehicle

            # Performance and cost criteria weights
            f'{ns}.Eval.FakeCarModel.performance_weights': {
                'top_speed': 0.2, 'acceleration': 0.3, 'range': 0.3, 'efficiency': 0.2
            },
            f'{ns}.Eval.FakeCarModel.cost_weights': {
                'manufacturing_cost': 0.5, 'maintenance_cost': 0.25, 'environmental_impact': 0.25
            },

            # ValueAnalysis configuration
            # (automatically receives dictionary outputs from DoE via shared visibility)
            f'{ns}.ValueAnalysis.criteria_weights': [0.4, 0.4, 0.2],  # [top_speed, range, cost]
        }
        return config_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
