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
from sos_trades_core.study_manager.study_manager import StudyManager
import pandas as pd
import numpy as np

from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory

from os.path import join, dirname


class Study(StudyManager):

    def __init__(self, run_usecase=True, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):

        self.uncertainty_quantification = 'UncertaintyQuantification'

        self.data_dir = join(dirname(__file__), 'data')
        self.data_df = pd.read_csv(
            join(self.data_dir, 'data_df.csv'))

        self.samples_dataframe = pd.read_csv(
            join(self.data_dir, 'samples_df.csv'))

        dict_values = {
            f'{self.study_name}.{self.uncertainty_quantification}.samples_df': self.samples_dataframe,
            f'{self.study_name}.{self.uncertainty_quantification}.data_df': self.data_df,
        }

        return dict_values


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    print("DONE")

    # # display all post_proc
    # post_processing_factory = PostProcessingFactory()
    # all_post_processings = post_processing_factory.get_all_post_processings(
    #     uc_cls.execution_engine, False, as_json=False, for_test=False)
    # for namespace, post_proc_list in all_post_processings.items():
    #     for chart in post_proc_list:
    #         for fig in chart.post_processings:
    #             fig.to_plotly().show()
