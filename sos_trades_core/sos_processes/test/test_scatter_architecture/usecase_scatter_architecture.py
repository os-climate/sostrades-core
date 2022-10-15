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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from sos_trades_core.study_manager.study_manager import StudyManager
import pandas as pd


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):

        activ_df = pd.DataFrame({'AC_list': ['NSA-300', 'NSA-300', 'NSA-400', 'NSA-400'],
                                 'subsystem_list': ['Airframe', 'Propulsion', 'Airframe', 'Propulsion'],
                                 'OSS': [True, True, True, True],
                                 'FHS': [True, True, True, True],
                                 'Pool': [True, True, True, True],
                                 'TSP': [True, True, True, True],
                                 'Sales': [True, True, True, True]})

        values_dict = {
            f'{self.study_name}.Business.actors_list': ['Airbus'],
            f'{self.study_name}.Business.Airbus.activation_df': activ_df}

        return values_dict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()

    print(uc_cls.execution_engine.dm.get_value(
        'usecase_scatter_architecture.Business.Airbus.activation_df'))
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