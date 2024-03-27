'''
Copyright 2024 Capgemini

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
from sostrades_core.datasets.dataset_mapping import DatasetsMapping
from sostrades_core.study_manager.study_manager import StudyManager

class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        return []

    def get_dataset_mapping(self):
        # Get dataset file
        datasets_file = __file__.replace(".py", ".json")
        # Deserialize it
        return DatasetsMapping.from_json_file(datasets_file)
