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
import logging
from sostrades_core.datasets.dataset_info.dataset_info_versions import (VERSION_V0, VERSION_V1)

from sostrades_core.datasets.datasets_connectors.abstract_multiversion_datasets_connector import (
    AbstractMultiVersionDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.json_datasets_connector.json_datasets_connectorV0 import (
    JSONDatasetsConnectorV0
)
from sostrades_core.datasets.datasets_connectors.json_datasets_connector.json_datasets_connectorV1 import (
    JSONDatasetsConnectorV1
)
from sostrades_core.datasets.datasets_serializers.datasets_serializer_factory import (
    DatasetSerializerType,
)


class JSONDatasetsConnectorMV(AbstractMultiVersionDatasetsConnector):  # FIXME: remove the MV when all is tested
    """
    Specific multi-version dataset connector for datasets in JSON format
    """
    __logger = logging.getLogger(__name__)
    VERSION_TO_CLASS = {VERSION_V0: JSONDatasetsConnectorV0,
                        VERSION_V1: JSONDatasetsConnectorV1}

    def __init__(self,
                 connector_id: str,
                 file_path: str,
                 create_if_not_exists: bool = False,
                 serializer_type: DatasetSerializerType = DatasetSerializerType.JSON):
        """
        Multi-version constructor with the instantiation arguments of connectors of type JSON.

        Args:
            connector_id (str): The identifier for the connector.
            file_path (str): The file path for this dataset connector.
            create_if_not_exists (bool, optional): Whether to create the file if it does not exist. Defaults to False.
            serializer_type (DatasetSerializerType, optional): The type of serializer to deserialize data from the connector. Defaults to DatasetSerializerType.JSON.
        """
        super().__init__(connector_id=connector_id,
                         file_path=file_path,
                         create_if_not_exists=create_if_not_exists,
                         serializer_type=serializer_type)

    def clear_dataset(self, dataset_id: str) -> None:
        """
        Utility method to remove the directory corresponding to a given dataset_id within the JSON database.

        Args:
            dataset_id (str): Identifier of the dataset to be removed
        """
        if self.__json_data is None:
            self.__json_data = self.__json_tooling.load_json_data()
        if dataset_id in self.__json_data:
            del self.__json_data[dataset_id]
        self.__json_tooling.save_json_data(self.__json_data)