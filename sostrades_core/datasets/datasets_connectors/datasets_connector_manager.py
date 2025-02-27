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
import logging
import os
import re

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetGenericException,
)
from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import (
    DatasetsConnectorFactory,
)
from sostrades_core.tools.metaclasses.no_instance import NoInstanceMeta


class DatasetConnectorNotFoundException(DatasetGenericException):
    """Exception when a dataset connector is not found"""

    def __init__(self, connector_name: str):
        self.connector_name = connector_name
        super().__init__(f"Dataset connector '{connector_name}' not found")


class DatasetsConnectorManager(metaclass=NoInstanceMeta):
    """Datasets connector manager"""

    CONNECTOR_TYPE_STR = "connector_type"
    CONNECTOR_IDENTIFIER_STR = AbstractDatasetsConnector.CONNECTOR_ID
    CONNECTOR_ARGS_STR = "connector_args"
    CONNECTOR_DEFAULT_REPOSITORY_RE = r'(?<=repos:)[\w-]+'
    CONNECTOR_DEFAULT_REPOSITORY_CLASS_PATH = "sostrades_core.datasets.datasets_connectors." \
                                              "local_repository_datasets_connector.LocalRepositoryDatasetsConnector"
    __registered_connectors = {}
    __logger = logging.getLogger(__name__)

    @classmethod
    def get_connector(cls, connector_identifier: str) -> AbstractDatasetsConnector:
        """
        Gets a connector given its identifier

        Args:
            connector_identifier (str): identifier of the connector

        Returns:
            AbstractDatasetsConnector: The requested connector

        Raises:
            DatasetConnectorNotFoundException: If the connector is not found

        """
        cls.__logger.debug(f"Getting connector {connector_identifier}")
        if connector_identifier not in cls.__registered_connectors:
            # check if the connector starts
            match = re.search(DatasetsConnectorManager.CONNECTOR_DEFAULT_REPOSITORY_RE, connector_identifier)
            if match:
                module_name = match.group(0)
                return DatasetsConnectorManager.register_connector(
                    connector_identifier=connector_identifier,
                    connector_type=cls.CONNECTOR_DEFAULT_REPOSITORY_CLASS_PATH,
                    module_name=module_name)
            else:
                raise DatasetConnectorNotFoundException(connector_identifier)
        return cls.__registered_connectors[connector_identifier]

    @classmethod
    def register_connector(cls, connector_identifier: str, connector_type: str, **connector_instanciation_fields
                           ) -> AbstractDatasetsConnector:
        """
        Register a connector with connector_instanciation_fields

        Args:
            connector_identifier (str): An unique name to identify clearly this connector
            connector_type (str): Class path to an existing connector (module_path.ClassName)
            **connector_instanciation_fields: dictionary of key/value needed by the connector

        Returns:
            AbstractDatasetsConnector: The registered connector

        """
        cls.__logger.debug(f"Registering connector {connector_identifier}")
        if connector_identifier in cls.__registered_connectors.keys():
            cls.__logger.debug(f'Existing connector \"{connector_identifier}\" is updated')

        connector = DatasetsConnectorFactory.get_connector(connector_identifier=connector_identifier,
                                                           connector_type=connector_type,
                                                           **connector_instanciation_fields)
        cls.__registered_connectors[connector_identifier] = connector
        return connector

    @classmethod
    def instanciate_connectors_from_json_file(cls, file_path: str) -> None:
        """
        Instantiates connectors from a JSON file

        Args:
            file_path (str): Path to the JSON file

        """
        with open(file=file_path, mode="r", encoding="utf-8") as file:
            json_data = json.load(file)

        for connector_data in json_data:
            connector_id = connector_data[cls.CONNECTOR_IDENTIFIER_STR]
            connector_type = connector_data[cls.CONNECTOR_TYPE_STR]
            cls.register_connector(connector_identifier=connector_id, connector_type=connector_type,
                                   **connector_data[cls.CONNECTOR_ARGS_STR])

# Initialize some sample connectors


default_sample_connector_file_path = os.path.join(os.path.dirname(__file__), "sample_connectors.json")
sample_connector_file_path = os.environ.get('SOS_TRADES_DATASET_CONNECTOR_CONFIGURATION',
                                            default_sample_connector_file_path)
if os.path.exists(sample_connector_file_path):
    DatasetsConnectorManager.instanciate_connectors_from_json_file(sample_connector_file_path)
else:
    logging.getLogger(__name__).warning(
        "Dataset connector sample connector file environment variable not set, using default file.")
    DatasetsConnectorManager.instanciate_connectors_from_json_file(default_sample_connector_file_path)
