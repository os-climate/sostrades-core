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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from sostrades_core.study_manager.study_manager import StudyManager
import pandas as pd


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        setup_data_list = []

        scenario_df = pd.DataFrame({'selected_scenario': [True, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_2']})
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': scenario_df}

        activation_df_1 = pd.DataFrame({'Business': ['Remy'],
                                        'CAPEX': [True],
                                        'OPEX': [False]})

        activation_df_2 = pd.DataFrame({'Business': ['Remy'],
                                        'CAPEX': [True],
                                        'OPEX': [True]})

        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.Business.activation_df'] = activation_df_1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.Business.activation_df'] = activation_df_2

        setup_data_list.append(dict_values)
        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()

#     ppf = PostProcessingFactory()
#     for disc in uc_cls.execution_engine.root_process.sos_disciplines:
#         if disc.sos_name == 'Airbus':
#             filters = ppf.get_post_processing_filters_by_discipline(
#                 disc)
#             graph_list = ppf.get_post_processing_by_discipline(
#                 disc, filters, as_json=False)
#
#             for graph in graph_list:
#                 graph.to_plotly()
