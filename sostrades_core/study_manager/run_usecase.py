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
import importlib
import logging
import os
import sys
from typing import Optional

import sostrades_core.study_manager.run_usecase
from sostrades_core.datasets.dataset_mapping import DatasetsMapping
from sostrades_core.study_manager.study_manager import StudyManager


def test_module_importability(module_name:str):
    """
    Tests if a module can be imported

    :param module_name: Dataset mapping file to use
    :type module_name: str
    """
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        raise Exception(f"Unable to import process module '{module_name}' is this module correct and in PYTHONPATH ?") from e

def run_usecase(usecase_file:str, dataset_mapping_json_file:Optional[str]):
    """
    Runs the usecase

    :param usecase_file: Usecase file
    :type usecase_file: str

    :param dataset_mapping_json_file: Dataset mapping file to use
    :type dataset_mapping_json_file: str
    """
    # Set logger level for datasets
    logging.getLogger("sostrades_core.datasets").setLevel(logging.DEBUG)
    # Test inputs
    if not os.path.exists(usecase_file):
        raise FileNotFoundError(f"File {usecase_file} does not exist")
    dataset_mapping = None
    if dataset_mapping_json_file is not None:
        if not os.path.exists(dataset_mapping_json_file):
            raise FileNotFoundError(f"File {dataset_mapping_json_file} does not exist")
        
        # Load process name
        dataset_mapping = DatasetsMapping.from_json_file(dataset_mapping_json_file)
        process_module_name = dataset_mapping.process_module_path

        test_module_importability(process_module_name + ".process")
    
    uc_cls = StudyManager(file_path=usecase_file)

    if dataset_mapping is not None:
        uc_cls.update_data_from_dataset_mapping(dataset_mapping_json_file)
    else:
        uc_cls.load_data()

    uc_cls.run()

if __name__ == "__main__":
    """
    Run a usecase from CLI
    Usage: python -m sostrades_core.study_manager.run_usecase <usecase_file> Optional<dataset_mapping_json_file>
    example
    python -m sostrades_core.study_manager.run_usecase ./sostrades_core/sos_processes/test/test_disc1_disc2_dataset/usecase_dataset.py ./sostrades_core/sos_processes/test/test_disc1_disc2_dataset/usecase_2datasets.json
    """
    if not 2<= len(sys.argv) <= 3:
        print(f"Usage: python -m {sostrades_core.study_manager.run_usecase.__name__} <usecase_file> Optional<dataset_mapping_json_file>")
        sys.exit(1)

    # Extract command-line arguments
    usecase_file = sys.argv[1]
    if (len(sys.argv) > 2):
        dataset_mapping_json_file = sys.argv[2]
    else:
        dataset_mapping_json_file = None

    # Call the main function with the provided arguments
    run_usecase(usecase_file=usecase_file, dataset_mapping_json_file=dataset_mapping_json_file)
