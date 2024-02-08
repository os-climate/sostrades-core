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
from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        setup_data_list = []

        scenario_df_outer = pd.DataFrame({'selected_scenario': [True, True],
                                          'scenario_name': ['sc1_business',
                                                            'sc2_business']})
        scenario_df_inner1 = pd.DataFrame({'selected_scenario': [True, True, False],
                                           'scenario_name': ['sc1_local_prod',
                                                             'sc2_abroad_prod',
                                                             'sc3_all_by_road']})
        scenario_df_inner2 = pd.DataFrame({'selected_scenario': [True, False, True],
                                           'scenario_name': ['sc1_local_prod',
                                                             'sc2_abroad_prod',
                                                             'sc3_all_by_road']})

        dict_values = {
                       f'{self.study_name}.outer_ms.samples_df': scenario_df_outer,
                       f'{self.study_name}.outer_ms.sc1_business.inner_ms.samples_df': scenario_df_inner1,
                       f'{self.study_name}.outer_ms.sc2_business.inner_ms.samples_df': scenario_df_inner2}


        activation_df_business_1 = pd.DataFrame({'Business': ['Remy'],
                                                 'CAPEX': [True],
                                                 'OPEX': [False]})

        activation_df_business_2 = pd.DataFrame({'Business': ['Remy'],
                                                 'CAPEX': [True],
                                                 'OPEX': [True]})

        activation_df_production_local = pd.DataFrame({'Production': ['Abroad','Local'],
                                                       'Road': [False, True],
                                                       'Plane': [False, False]})
        activation_df_production_abroad = pd.DataFrame({'Production': ['Abroad','Local'],
                                                         'Road': [True, False],
                                                         'Plane': [True, False]})
        activation_df_production_road = pd.DataFrame({'Production': ['Abroad','Local'],
                                                      'Road': [True, True],
                                                      'Plane': [False, False]})
        
        # setup the business architectures
        dict_values[f'{self.study_name}.outer_ms.sc1_business.inner_ms.sc1_local_prod.Business.activation_df'] = \
            activation_df_business_1
        dict_values[f'{self.study_name}.outer_ms.sc1_business.inner_ms.sc2_abroad_prod.Business.activation_df'] = \
            activation_df_business_1
        dict_values[f'{self.study_name}.outer_ms.sc2_business.inner_ms.sc1_local_prod.Business.activation_df'] = \
            activation_df_business_2
        dict_values[f'{self.study_name}.outer_ms.sc2_business.inner_ms.sc3_all_by_road.Business.activation_df'] = \
            activation_df_business_2

        # setup the production architectures
        dict_values[f'{self.study_name}.outer_ms.sc1_business.inner_ms.sc1_local_prod.Production.activation_df'] = \
            activation_df_production_local
        dict_values[f'{self.study_name}.outer_ms.sc1_business.inner_ms.sc2_abroad_prod.Production.activation_df'] = \
            activation_df_production_abroad
        dict_values[f'{self.study_name}.outer_ms.sc2_business.inner_ms.sc1_local_prod.Production.activation_df'] = \
            activation_df_production_local
        dict_values[f'{self.study_name}.outer_ms.sc2_business.inner_ms.sc3_all_by_road.Production.activation_df'] = \
            activation_df_production_road

        setup_data_list.append(dict_values)
        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
