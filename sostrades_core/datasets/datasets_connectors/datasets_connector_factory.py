"""
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
"""
from enum import Enum
from sostrades_core.datasets.datasets_connectors.json_datasets_connector import (
    JSONDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
)
from sostrades_core.tools.metaclasses.no_instance import NoInstanceMeta


class DatasetConnectorTypes(Enum):
    """
    Dataset connector types anum
    """

    JSON = JSONDatasetsConnector


class DatasetsConnectorFactory(metaclass=NoInstanceMeta):
    """
    Dataset connector factory
    """

    @staticmethod
    def get_connector(
        connector_type: DatasetConnectorTypes, **connector_instanciation_fields
    ) -> AbstractDatasetsConnector:
        """
        Instanciate a connector of type connector_type with provided arguments
        Raises ValueError if type is invalid

        :param connector_type: connector type to instanciate
        :type connector_type: DatasetConnectorTypes
        """
        if not isinstance(connector_type, DatasetConnectorTypes) or not issubclass(
            connector_type.value, AbstractDatasetsConnector
        ):
            raise ValueError(f"Unexpected connector type {connector_type}")
        return connector_type.value(**connector_instanciation_fields)


if __name__ == "__main__":
    """
    Example usage of the factory"""
    DatasetsConnectorFactory.get_connector(DatasetConnectorTypes.JSON, filename="aa")

    connector_instanciation_dict = {"filename": "test_filename.json"}
    DatasetsConnectorFactory.get_connector(
        DatasetConnectorTypes.JSON, **connector_instanciation_dict
    )
