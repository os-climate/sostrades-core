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
import logging
import random
import string
import re
from typing import Any, List

from arango import ArangoClient, CollectionListError
from arango.collection import StandardCollection
import numpy as np
import pandas as pd

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetGenericException,
    DatasetNotFoundException,
    DatasetUnableToInitializeConnectorException,
)
from sostrades_core.datasets.datasets_serializers.datasets_serializer_factory import DatasetSerializerType, DatasetsSerializerFactory
from sostrades_core.datasets.datasets_serializers.json_datasets_serializer import JSONDatasetsSerializer


class ArangoDatasetsConnector(AbstractDatasetsConnector):
    """
    Specific dataset connector for dataset in arango db format
    """
    COLLECTION_NAME_STR = "name"
    COLLECTION_SYSTEM_STR = "system"
    VALUE_STR = "value"
    KEY_STR = "_key"
    DATASET_NAME_STR = "dataset_name"
    MAX_KEY_SIZE = 254

    def __init__(self, host: str, db_name: str, username: str, password: str, 
                 serializer_type:DatasetSerializerType=DatasetSerializerType.JSON, 
                 datasets_descriptor_collection_name:str="datasets"):
        """
        Constructor for Arango data connector

        :param host: Host to connect to
        :type host: str

        :param db_name: Database name
        :type db_name: str

        :param username: Username
        :type username: str

        :param password: Password
        :type password: str

        :param serializer_type: type of serializer to deserialize data from connector
        :type serializer_type: DatasetSerializerType (JSON for jsonDatasetSerializer)

        :param datasets_descriptor_collection_name: Database describing datasets
        :type datasets_descriptor_collection_name: str
        """
        super().__init__()
        self.__logger = logging.getLogger(__name__)
        self.__logger.debug("Initializing Arango connector")
        self.__datasets_serializer = DatasetsSerializerFactory.get_serializer(serializer_type)

        # Connect to database
        try:
            client = ArangoClient(hosts=host)
            self.db = client.db(name=db_name, username=username, password=password)

            if not self.db.has_collection(name=datasets_descriptor_collection_name):
                raise Exception(f"Expected to find collection {datasets_descriptor_collection_name} describing datasets")
            self.datasets_descriptor_collection_name = datasets_descriptor_collection_name
            
        except Exception as exc:
            raise DatasetUnableToInitializeConnectorException(connector_type=ArangoDatasetsConnector) from exc

    def __name_to_valid_arango_collection_name(self, dataset_name:str) -> str:
        """
        Constructor for Arango data connector
        Checks that this collection does not exist

        :param dataset_name: dataset name to clean
        :type dataset_name: str
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
        """
        collection = self.db.collection(name=self.datasets_descriptor_collection_name)
        return {document[ArangoDatasetsConnector.DATASET_NAME_STR]:document[ArangoDatasetsConnector.KEY_STR] for document in collection}


    def __get_dataset_collection(self, name: str) -> StandardCollection:
        """
        Get the collection associated with datasets

        :param dataset_identifier: identifier of the dataset
        :type dataset_identifier: str

        :param data_to_get: data to retrieve, list of names
        :type data_to_get: List[str]
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

    def get_values(self, dataset_identifier: str, data_to_get: dict[str:str]) -> None:
        """
        Method to retrieve data from JSON and fill a data_dict

        :param dataset_identifier: identifier of the dataset
        :type dataset_identifier: str

        :param data_to_get: data to retrieve, dict of names and types
        :type data_to_get: dict[str:str]
        """
        self.__logger.debug(f"Getting values {data_to_get.keys()} for dataset {dataset_identifier} for connector {self}")
        dataset_collection = self.__get_dataset_collection(name=dataset_identifier)

        # Retrieve the data
        cursor = dataset_collection.get_many(data_to_get.keys())

        # Process the results
        result_data = {doc[ArangoDatasetsConnector.KEY_STR]: 
                        self.__datasets_serializer.convert_from_dataset_data(doc[ArangoDatasetsConnector.KEY_STR], 
                                                        doc[ArangoDatasetsConnector.VALUE_STR], 
                                                        data_to_get)
                        for doc in cursor}
        self.__logger.debug(
            f"Values obtained {list(result_data.keys())} for dataset {dataset_identifier} for connector {self}"
        )
        return result_data

    def write_values(self, dataset_identifier: str, values_to_write: dict[str:Any], data_types_dict: dict[str:str]) -> None:
        """
        Method to write data

        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: Dict[str:Any]
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        """
        self.__logger.debug(f"Writing values in dataset {dataset_identifier} for connector {self}")
        dataset_collection = self.__get_dataset_collection(name=dataset_identifier)
        # prepare query to write
        data_for_arango = [{ArangoDatasetsConnector.KEY_STR: tag, 
                            ArangoDatasetsConnector.VALUE_STR: self.__datasets_serializer.convert_to_dataset_data(tag, value, data_types_dict)}
                            for tag, value in values_to_write.items()]

        # Write items
        dataset_collection.insert_many(data_for_arango, overwrite=True)

    def get_values_all(self, dataset_identifier: str, data_types_dict:dict[str:str]) -> dict[str:Any]:
        """
        Get all values from a dataset for Arango
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        """
        self.__logger.debug(f"Getting all values for dataset {dataset_identifier} for connector {self}")
        dataset_collection = self.__get_dataset_collection(name=dataset_identifier)

        # Process all data
        result_data = {doc[ArangoDatasetsConnector.KEY_STR]: self.__datasets_serializer.convert_from_dataset_data(doc[ArangoDatasetsConnector.KEY_STR], 
                                                                                            doc[ArangoDatasetsConnector.VALUE_STR], 
                                                                                            data_types_dict) 
                        for doc in dataset_collection}
        return result_data
    
    def get_datasets_available(self) -> list[str]:
        """
        Get all available datasets for a specific API
        """
        self.__logger.debug(f"Getting all datasets for connector {self}")
        mapping = self.__parse_datasets_mapping()
        return list(mapping.keys())

    def write_dataset(
        self,
        dataset_identifier: str,
        values_to_write: dict[str:Any],
        data_types_dict:dict[str:str],
        create_if_not_exists: bool = True,
        override: bool = False,
    ) -> None:
        """
        Write a dataset from Arango
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str:Any]
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        """
        self.__logger.debug(
            f"Writing dataset {dataset_identifier} for connector {self} (override={override}, create_if_not_exists={create_if_not_exists})"
        )
        # Check if dataset exists
        mapping = self.__parse_datasets_mapping()
        if dataset_identifier not in mapping:
            # Handle dataset creation
            if create_if_not_exists:
                # Generate a dataset uid
                dataset_uid = self.__name_to_valid_arango_collection_name(dataset_identifier)
                
                # Create matching collection
                collection = self.db.collection(name=self.datasets_descriptor_collection_name)
                collection.insert({ArangoDatasetsConnector.KEY_STR: dataset_uid, ArangoDatasetsConnector.DATASET_NAME_STR: dataset_identifier}, overwrite=True)
                self.db.create_collection(name=dataset_uid)
            else:
                raise DatasetNotFoundException(dataset_identifier)
        else:
            # Handle override
            if not override:
                raise DatasetGenericException(f"Dataset {dataset_identifier} would be overriden")
            if not self.db.has_collection(name=mapping[dataset_identifier]):
                self.db.create_collection(name=mapping[dataset_identifier])

        self.write_values(dataset_identifier=dataset_identifier, values_to_write=values_to_write, data_types_dict=data_types_dict)

    

if __name__ == "__main__":
    """
    Example usage using docker deployment for arango
    """
    
    logging.getLogger("sostrades_core.datasets").setLevel(logging.DEBUG)
    connector_values = {
        "host": "http://127.0.0.1:8529",
        "db_name": "os-climate",
        "username": "root",
        "password": "ArangoDB_BfPM",
    }

    connector = ArangoDatasetsConnector(**connector_values)
    # Write values
    connector.write_values(dataset_identifier="test_dataset_collection", values_to_write={"x": 1, "y": "str_y2"})

    # Read values
    print(connector.get_values(dataset_identifier="test_dataset_collection", data_to_get={"x":'int', "y":'string'}))

    # Read dataset
    print(connector.get_values_all(dataset_identifier="test_dataset_collection"), data_to_get={"x":'int', "y":'string'})

    # Write dataset
    #connector.write_dataset(dataset_identifier="test_dataset_collection_2", values_to_write={"x": 1, "y": "str_y2"})
    print(connector.get_datasets_available())
