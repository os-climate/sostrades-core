'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16-2024/06/28 Copyright 2024 Capgemini

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
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        setup_data_list = []
        # private values AC model
        private_values = {
            self.study_name + '.x': 10.,
            self.study_name + '.Disc1.a': 5.,
            self.study_name + '.Disc1.b': 25431.,
            self.study_name + '.y': 4.,
            self.study_name + '.Disc2.constant': 3.1416,
            self.study_name + '.Disc2.power': 2}
        setup_data_list.append(private_values)
        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
    ppf = PostProcessingFactory()

    all_post_processings = ppf.get_all_post_processings(uc_cls.execution_engine, False, as_json=False, for_test=False)

    for post_proc_list in all_post_processings.values():
        for chart in post_proc_list:
            for fig in chart.post_processings:
                fig.to_plotly().show()
