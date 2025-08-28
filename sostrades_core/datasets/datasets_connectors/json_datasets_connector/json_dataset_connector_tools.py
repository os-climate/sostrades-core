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

import json
import os

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import DatasetGenericException
from sostrades_core.tools.folder_operations import makedirs_safe


class JSONDatasetsConnectorTools():

    def __init__(self, file_path: str, create_if_not_exists: bool) -> None:
        """
        Initialize the JSONDatasetsConnectorTools.

        Args:
            file_path (str): The path to the JSON file.
            create_if_not_exists (bool): Whether to create the file if it does not exist.

        """
        self.__file_path = file_path
        # create file if not exist
        if create_if_not_exists and not os.path.exists(file_path):
            makedirs_safe(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def load_json_data(self) -> dict:
        """
        Load data from the JSON file.

        Returns:
            dict: The loaded JSON data.

        Raises:
            DatasetGenericException: If the JSON file is not found.

        """
        json_data = {}
        db_path = self.__file_path
        if not os.path.exists(db_path):
            raise DatasetGenericException(f"The connector json file is not found at {db_path}") from FileNotFoundError()

        with open(db_path, "r", encoding="utf-8") as file:
            json_data = json.load(fp=file)

        return json_data

    def save_json_data(self, json_data: dict) -> None:
        """
        Save data to the JSON file.

        Args:
            json_data (dict): The data to save.

        Raises:
            DatasetGenericException: If the JSON file is not found.

        """
        db_path = self.__file_path
        if not os.path.exists(db_path):
            raise DatasetGenericException() from FileNotFoundError(f"The connector json file is not found at {db_path}")

        with open(db_path, "w", encoding="utf-8") as file:
            json.dump(obj=json_data, fp=file, indent=4)
