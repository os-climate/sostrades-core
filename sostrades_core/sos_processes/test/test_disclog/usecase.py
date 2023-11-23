'''
Copyright 2023 Capgemini

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
from sostrades_core.study_manager.study_manager import StudyManager
import time


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):

        dict_values = {
            'usecase.DiscLog.log_lines': 10,
            'usecase.DiscLog.wait_time_s': 5,
            }
        return dict_values


if '__main__' == __name__:
    start = time.time()
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
    stop = time.time()
    print(stop - start)
