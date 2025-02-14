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
from __future__ import annotations

import logging
import random
import re
import string
from typing import TYPE_CHECKING, Any

from arango import ArangoClient, CollectionListError

from sostrades_core.datasets.dataset_info.dataset_info_v0 import DatasetInfoV0
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetGenericException,
    DatasetNotFoundException,
    DatasetUnableToInitializeConnectorException,
)
from sostrades_core.datasets.datasets_serializers.datasets_serializer_factory import (
    DatasetSerializerType,
    DatasetsSerializerFactory,
)

if TYPE_CHECKING:
    from arango.collection import StandardCollection

    from sostrades_core.datasets.dataset_info.abstract_dataset_info import AbstractDatasetInfo


class ArangoDatasetsConnector(AbstractDatasetsConnector):
    """Specific dataset connector for dataset in arango db format"""

    COLLECTION_NAME_STR = "name"
    COLLECTION_SYSTEM_STR = "system"
    VALUE_STR = "value"
    KEY_STR = "_key"
    DATASET_NAME_STR = "dataset_name"
    MAX_KEY_SIZE = 254

    def __init__(self, connector_id: str, host: str, db_name: str, username: str, password: str,
                 serializer_type: DatasetSerializerType = DatasetSerializerType.JSON,
                 datasets_descriptor_collection_name: str = "datasets") -> None:
        """
        Constructor for Arango data connector

        Args:
            connector_id (str): Connector identifier
            host (str): Host to connect to
            db_name (str): Database name
            username (str): Username
            password (str): Password
            serializer_type (DatasetSerializerType, optional): Type of serializer to deserialize data from connector. Defaults to DatasetSerializerType.JSON.
            datasets_descriptor_collection_name (str, optional): Database describing datasets. Defaults to "datasets".

        """
        super().__init__()
        self.__logger = logging.getLogger(__name__)
        self.__logger.debug("Initializing Arango connector")
        self.__datasets_serializer = DatasetsSerializerFactory.get_serializer(serializer_type)
        self.connector_id = connector_id

        # Connect to database
        try:
            client = ArangoClient(hosts=host)
            self.db = client.db(name=db_name, username=username, password=password)

            if not self.db.has_collection(name=datasets_descriptor_collection_name):
                raise DatasetGenericException(f"Expected to find collection {datasets_descriptor_collection_name} describing datasets")
            self.datasets_descriptor_collection_name = datasets_descriptor_collection_name

        except Exception as exc:
            raise DatasetUnableToInitializeConnectorException(connector_type=ArangoDatasetsConnector) from exc

    def __name_to_valid_arango_collection_name(self, dataset_name: str) -> str:
        """
        Converts a dataset name to a valid ArangoDB collection name.
        Checks that this collection does not exist.

        Args:
            dataset_name (str): Dataset name to clean

        Returns:
            str: Valid ArangoDB collection name

        """
        # Remove characters not allowed in collection names
        filtered_name = re.sub(r'[^a-zA-Z0-9_\-]', '', dataset_name)

        # Ensure the name starts with a letter
        if filtered_name and not filtered_name[0].isalpha():
            filtered_name = 'D' + filtered_name

        # Ensure the name does not exceed the maximum length
        max_length = 256
        filtered_name = filtered_name[:max_length]

        if len(filtered_name) == 0 or self.db.has_collection(name=filtered_name):
            # generate a random id
            # need only alpha characters
            return ''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(ArangoDatasetsConnector.MAX_KEY_SIZE))

        return filtered_name

    def __parse_datasets_mapping(self) -> dict[str, StandardCollection]:
        """
        Gets the mapping between datasets name and datasets collections

        Returns:
            dict[str, StandardCollection]: Mapping between dataset names and collections

        """
        collection = self.db.collection(name=self.datasets_descriptor_collection_name)
        return {document[ArangoDatasetsConnector.DATASET_NAME_STR]: document[ArangoDatasetsConnector.KEY_STR] for document in collection}

    def __get_dataset_collection(self, name: str) -> StandardCollection:
        """
        Get the collection associated with a dataset.

        Args:
            name (str): Dataset name

        Returns:
            StandardCollection: Collection associated with the dataset

        Raises:
            DatasetNotFoundException: If the dataset is not found

        """
        try:
            mapping = self.__parse_datasets_mapping()
            dataset_collection_name = mapping[name]
            if not self.db.has_collection(name=dataset_collection_name):
                raise DatasetNotFoundException(dataset_name=name)
            return self.db.collection(name=dataset_collection_name)
        except CollectionListError as exc:
            raise DatasetNotFoundException(dataset_name=name) from exc
        except KeyError as exc:
            raise DatasetNotFoundException(dataset_name=name) from exc

    def _get_values(self, dataset_identifier: AbstractDatasetInfo, data_to_get: dict[str, str]) -> dict[str, Any]:
        """
        Method to retrieve data from JSON and fill a data_dict

        Args:
            dataset_identifier (AbstractDatasetInfo): Identifier of the dataset
            data_to_get (dict[str, str]): Data to retrieve, dict of names and types

        Returns:
            dict[str, Any]: Retrieved data

        """
        self.__logger.debug(f"Getting values {data_to_get.keys()} for dataset {dataset_identifier.dataset_id} for connector {self}")
        dataset_collection = self.__get_dataset_collection(name=dataset_identifier.dataset_id)

        # Retrieve the data
        cursor = dataset_collection.get_many(data_to_get.keys())

        # Process the results
        result_data = {doc[ArangoDatasetsConnector.KEY_STR]:
                        self.__datasets_serializer.convert_from_dataset_data(doc[ArangoDatasetsConnector.KEY_STR],
                                                        doc[ArangoDatasetsConnector.VALUE_STR],
                                                        data_to_get)
                        for doc in cursor}
        self.__logger.debug(
            f"Values obtained {list(result_data.keys())} for dataset {dataset_identifier.dataset_id} for connector {self}"
        )
        return result_data

    def _write_values(self, dataset_identifier: AbstractDatasetInfo, values_to_write: dict[str, Any], data_types_dict: dict[str, str]) -> dict[str, Any]:
        """
        Method to write data

        Args:
            dataset_identifier (AbstractDatasetInfo): Dataset identifier for connector
            values_to_write (dict[str, Any]): Dict of data to write {name: value}
            data_types_dict (dict[str, str]): Dict of data type {name: type}

        Returns:
            dict[str, Any]: Written values

        """
        self.__logger.debug(f"Writing values in dataset {dataset_identifier.dataset_id} for connector {self}")
        dataset_collection = self.__get_dataset_collection(name=dataset_identifier.dataset_id)
        # prepare query to write
        data_for_arango = [{ArangoDatasetsConnector.KEY_STR: tag,
                            ArangoDatasetsConnector.VALUE_STR: self.__datasets_serializer.convert_to_dataset_data(tag, value, data_types_dict)}
                            for tag, value in values_to_write.items()]

        # Write items
        dataset_collection.insert_many(data_for_arango, overwrite=True)
        return values_to_write

    def _get_values_all(self, dataset_identifier: AbstractDatasetInfo, data_types_dict: dict[str, str]) -> dict[str, Any]:
        """
        Get all values from a dataset for Arango

        Args:
            dataset_identifier (AbstractDatasetInfo): Dataset identifier for connector
            data_types_dict (dict[str, str]): Dict of data type {name: type}

        Returns:
            dict[str, Any]: All values from the dataset

        """
        self.__logger.debug(f"Getting all values for dataset {dataset_identifier.dataset_id} for connector {self}")
        dataset_collection = self.__get_dataset_collection(name=dataset_identifier.dataset_id)

        # Process all data
        result_data = {doc[ArangoDatasetsConnector.KEY_STR]: self.__datasets_serializer.convert_from_dataset_data(doc[ArangoDatasetsConnector.KEY_STR],
                                                                                            doc[ArangoDatasetsConnector.VALUE_STR],
                                                                                            data_types_dict)
                        for doc in dataset_collection}
        return result_data

    def get_datasets_available(self) -> list[AbstractDatasetInfo]:
        """
        Get all available datasets for a specific API

        Returns:
            list[AbstractDatasetInfo]: List of available datasets

        """
        self.__logger.debug(f"Getting all datasets for connector {self}")
        mapping = self.__parse_datasets_mapping()
        return [DatasetInfoV0(self.connector_id, dataset_id) for dataset_id in list(mapping.keys())]

    def _write_dataset(
        self,
        dataset_identifier: AbstractDatasetInfo,
        values_to_write: dict[str, Any],
        data_types_dict: dict[str, str],
        create_if_not_exists: bool = True,
        override: bool = False,
    ) -> dict[str, Any]:
        """
        Write a dataset from Arango

        Args:
            dataset_identifier (AbstractDatasetInfo): Dataset identifier for connector
            values_to_write (dict[str, Any]): Dict of data to write {name: value}
            data_types_dict (dict[str, str]): Dict of data types {name: type}
            create_if_not_exists (bool, optional): Create the dataset if it does not exists (raises otherwise). Defaults to True.
            override (bool, optional): Override dataset if it exists (raises otherwise). Defaults to False.

        Returns:
            dict[str, Any]: Written values

        """
        self.__logger.debug(
            f"Writing dataset {dataset_identifier.dataset_id} for connector {self} (override={override}, create_if_not_exists={create_if_not_exists})"
        )
        # Check if dataset exists
        mapping = self.__parse_datasets_mapping()
        if dataset_identifier.dataset_id not in mapping:
            # Handle dataset creation
            if create_if_not_exists:
                # Generate a dataset uid
                dataset_uid = self.__name_to_valid_arango_collection_name(dataset_identifier.dataset_id)

                # Create matching collection
                collection = self.db.collection(name=self.datasets_descriptor_collection_name)
                collection.insert({ArangoDatasetsConnector.KEY_STR: dataset_uid, ArangoDatasetsConnector.DATASET_NAME_STR: dataset_identifier}, overwrite=True)
                self.db.create_collection(name=dataset_uid)
            else:
                raise DatasetNotFoundException(dataset_identifier.dataset_id)
        else:
            # Handle override
            if not override:
                raise DatasetGenericException(f"Dataset {dataset_identifier.dataset_id} would be overriden")
            if not self.db.has_collection(name=mapping[dataset_identifier.dataset_id]):
                self.db.create_collection(name=mapping[dataset_identifier.dataset_id])

        return self.write_values(dataset_identifier=dataset_identifier, values_to_write=values_to_write, data_types_dict=data_types_dict)
