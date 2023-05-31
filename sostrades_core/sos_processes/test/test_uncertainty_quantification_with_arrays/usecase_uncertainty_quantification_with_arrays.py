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
from os.path import join, dirname

import numpy as np
import pandas as pd

# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory


class Study(StudyManager):

    def __init__(self, run_usecase=True, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        self.uncertainty_quantification = 'UncertaintyQuantification'

        self.data_dir = join(dirname(__file__), 'data')
        self.samples_dataframe = pd.read_csv(
            join(self.data_dir, 'samples_df.csv'))

        # ADDING ARRAYS in inputs
        x_range = np.arange(50, 150, 15)
        y_range = np.arange(40, 80, 8)
        z_range = np.arange(60, 120, 15)
        x, y, z = np.meshgrid(x_range, y_range, z_range)
        triplets = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        samples_dataframe = pd.concat([self.samples_dataframe[:-1]] * len(triplets))
        samples_dataframe = pd.concat([samples_dataframe, self.samples_dataframe.iloc[-1:]])
        array_var_column = []
        for triplet in triplets:
            array_var_column += [triplet] * len(self.samples_dataframe[:-1])
        array_var_column.append(10.)
        samples_dataframe['input_array'] = array_var_column
        samples_dataframe['scenario'] = [f'scenario_{i}' for i in range(len(samples_dataframe) - 1)] + ['reference']
        self.samples_dataframe = samples_dataframe

        np.random.seed(42)

        Var1 = np.random.uniform(-1, 1, size=len(self.samples_dataframe))
        Var2 = np.random.uniform(-1, 1, size=len(self.samples_dataframe))

        # set outputs to be both floats and arrays
        out1 = list(pd.Series(Var1 + Var2) * 100000)
        out_array = list(np.array([(Var1 * Var2) * 100_000,
                                   (Var1 ** 2 + Var2 ** 2) * 100_000,
                                   (Var1 ** 4 - Var2 ** 2) * 100_000,
                                   (-Var1 ** 2 - Var2 ** 2) * 100_000]).T)

        self.data_df = pd.DataFrame(
            {'scenario': self.samples_dataframe['scenario'], 'output1': out1, 'output_array': out_array})

        self.input_selection = pd.DataFrame({'selected_input': [True, True, True, True],
                           'full_name': ['COC', 'RC', 'NRC', 'input_array'],
                           'shortest_name': ['COC', 'RC', 'NRC', 'input_array']})

        self.output_selection = pd.DataFrame({'selected_output': [True, True],
                            'full_name': ['output1', 'output_array', ],
                            'shortest_name': ['output1', 'output_array', ]})

        dspace = pd.DataFrame({
            'shortest_name': ['COC', 'RC', 'NRC', 'input_array'],
            'lower_bnd': [85., 80., 80., np.array([50., 40., 60])],
            'upper_bnd': [105., 120., 120., np.array([150., 80., 120.])],
            'nb_points': [10, 10, 10, 10],
            'full_name': ['COC', 'RC', 'NRC', 'input_array'],
        })

        """
        data_details_df = pd.DataFrame({'type': ['input'] * 3 + ['output'] * 3,
                                        'variable': ['COC', 'RC', 'NRC', 'output1', 'output2', 'output3'],
                                        'name': ['COC', 'RC', 'NRC', 'output1', 'output2', 'output3'],
                                        'unit': '$'})
        """
        dict_values = {
            f'{self.study_name}.{self.uncertainty_quantification}.eval_inputs': self.input_selection,
            f'{self.study_name}.{self.uncertainty_quantification}.eval_outputs': self.output_selection,
            f'{self.study_name}.{self.uncertainty_quantification}.samples_inputs_df': self.samples_dataframe,
            f'{self.study_name}.{self.uncertainty_quantification}.samples_outputs_df': self.data_df,
            f'{self.study_name}.{self.uncertainty_quantification}.design_space': dspace,
            #f'{self.study_name}.{self.uncertainty_quantification}.data_details_df': data_details_df,

        }

        return dict_values


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    print("DONE")

    # # display all post_proc
    post_processing_factory = PostProcessingFactory()
    all_post_processings = post_processing_factory.get_all_post_processings(
        uc_cls.execution_engine, False, as_json=False, for_test=False)
    for namespace, post_proc_list in all_post_processings.items():
        for chart in post_proc_list:
            for fig in chart.post_processings:
                fig.to_plotly()  # .show()
