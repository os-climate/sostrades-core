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

import sostrades_core.study_manager.run_usecase
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.datasets.dataset_mapping import DatasetsMapping

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

def run_usecase(dataset_mapping_json_file:str):
    """
    Instanciate a connector of type connector_type with provided arguments
    Raises ValueError if type is invalid

    :param dataset_mapping_json_file: Dataset mapping file to use
    :type dataset_mapping_json_file: str
    """
    # Set logger level for datasets
    logging.getLogger("sostrades_core.datasets").setLevel(logging.DEBUG)
    # Test inputs
    if not os.path.exists(dataset_mapping_json_file):
        raise FileNotFoundError(f"File {dataset_mapping_json_file} does not exist")
    
    # Load process name
    dataset_mapping = DatasetsMapping.from_json_file(dataset_mapping_json_file)
    process_module_name = dataset_mapping.process_module_path

    test_module_importability(process_module_name + ".process")
    
    # dataset_mapping_json_file = ./sostrades_core/data/study_001_test.json
    # study_name => study_001_test
    study_name = ".".join(os.path.basename(dataset_mapping_json_file).split(".")[:-1])

    # TODO fix this, may not work with latest changes
    uc_cls = StudyManager(process_module_name=process_module_name, study_name=study_name)
    uc_cls.load_study(dataset_mapping_json_file)
    uc_cls.run()

if __name__ == "__main__":
    """
    Run a usecase from CLI
    Usage: python -m sostrades_core.study_manager.run_usecase <dataset_mapping_json_file>
    example
    python -m sostrades_core.study_manager.run_usecase ./platform/sostrades-core/sostrades_core/sos_processes/test/test_disc1_disc2_dataset/usecase_dataset.json
    """
    if len(sys.argv) != 2:
        print(f"Usage: python -m {sostrades_core.study_manager.run_usecase.__name__} <dataset_mapping_json_file>")
        sys.exit(1)

    # Extract command-line arguments
    dataset_mapping_json_file = sys.argv[1]

    # Call the main function with the provided arguments
    run_usecase(dataset_mapping_json_file=dataset_mapping_json_file)
