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
from enum import Enum
import logging
from sostrades_core.datasets.datasets_connectors.json_datasets_connector import (
    JSONDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.local_datasets_connector import (
    LocalDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.arango_datasets_connector import (
    ArangoDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetUnableToInitializeConnectorException,
)
from sostrades_core.datasets.datasets_connectors.sospickle_datasets_connector import SoSPickleDatasetsConnector
from sostrades_core.tools.metaclasses.no_instance import NoInstanceMeta


class DatasetConnectorType(Enum):
    """
    Dataset connector types anum
    """

    JSON = JSONDatasetsConnector
    Local = LocalDatasetsConnector
    Arango = ArangoDatasetsConnector
    SoSpickle = SoSPickleDatasetsConnector

    @classmethod
    def get_enum_value(cls, value_str):
        try:
            # Iterate through the enum members and find the one with a matching value
            return next(member for member in cls if member.name == value_str)
        except StopIteration:
            raise ValueError(f"No matching enum value found for '{value_str}'")


class DatasetsConnectorFactory(metaclass=NoInstanceMeta):
    """
    Dataset connector factory
    """
    __logger = logging.getLogger(__name__)

    @classmethod
    def get_connector(cls,
        connector_type: DatasetConnectorType, **connector_instanciation_fields
    ) -> AbstractDatasetsConnector:
        """
        Instanciate a connector of type connector_type with provided arguments
        Raises ValueError if type is invalid

        :param connector_type: connector type to instanciate
        :type connector_type: DatasetConnectorType
        """
        cls.__logger.debug(f"Instanciating connector of type {connector_type}")
        if not isinstance(connector_type, DatasetConnectorType) or not issubclass(
            connector_type.value, AbstractDatasetsConnector
        ):
            raise ValueError(f"Unexpected connector type {connector_type}")
        try:
            return connector_type.value(**connector_instanciation_fields)
        except TypeError as exc:
            raise DatasetUnableToInitializeConnectorException(connector_type) from exc
