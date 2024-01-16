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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from sostrades_core.datasets.datasets_connectors.json_datasets_connector import JSONDatasetsConnector
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
import time
from os.path import join, dirname
from os.path import abspath, basename, dirname, relpath, splitext, join, exists

class Study(StudyManager):

    def __init__(self, execution_engine=None):
        # get the process folder name
        file_path = __file__
        study_file_path = abspath(file_path)
        study_file_name = splitext(basename(study_file_path))[0]
        module_path = dirname(study_file_path)
        process_name = basename(module_path)

        # Find the module path
        module_path = dirname(module_path)
        module_path_list = []

        # Check if __init__.py exists in the parent directory
        # If yes, it is a module
        # If not, we stop
        while exists(join(module_path, '__init__.py')):
            module_path_list.append(basename(module_path))
            module_path = dirname(module_path)

        repository_name = '.'.join(module_path_list[::-1])
        super().__init__(repository_name,process_name, study_file_name, execution_engine=execution_engine)

    def setup_usecase(self):

        dict_values = {
            f'{self.study_name}.a': 1,
            f'{self.study_name}.Disc1.b': "StringInputDisc1",
            f'{self.study_name}.Disc2.b': "StringInputDisc2",
            f'{self.study_name}.Disc1.c': "CCCCCC11111",
            f'{self.study_name}.Disc2.c': "CCCCCC222222",
            f'{self.study_name}.Disc1VirtualNode.x': 4.,
            f'{self.study_name}.Disc2VirtualNode.x': 5.,
            }
        return dict_values


if '__main__' == __name__:
    # test study with only one dataset
    json_study_file_path = join(dirname(__file__), 'usecase_dataset.json')
    uc_cls = Study()
    uc_cls.load_study(json_study_file_path)
    uc_cls.run()

    # test with 2 datasets
    json_study_file_path = join(dirname(__file__), 'usecase_2datasets.json')
    uc_cls2 = Study()
    uc_cls2.load_study(json_study_file_path)
    uc_cls2.run()


