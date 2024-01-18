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
from typing import Any, List

from arango import ArangoClient, CollectionListError
from arango.collection import StandardCollection

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import AbstractDatasetsConnector, DatasetGenericException, DatasetNotFoundException, DatasetUnableToInitializeConnectorException


class ArangoDatasetsConnector(AbstractDatasetsConnector):
    """
    Specific dataset connector for dataset in arango db format
    """

    def __init__(self, host:str, db_name:str, username:str, password:str):
        """
        Constructor for Arango data connector

        :param hostname: Host to connect to
        :type hostname: str
        """
        super().__init__()

        # Connect to database
        try:
            client = ArangoClient(hosts=host)
            self.db = client.db(name=db_name, username=username, password=password)
            # Check if db exists and we can retrieve details
            self.db.details()
        except Exception as exc:
            raise DatasetUnableToInitializeConnectorException(connector_type=ArangoDatasetsConnector) from exc
        
    def __get_dataset_collection(self, name:str) -> StandardCollection:
        try:
            if not self.db.has_collection(name=name):
                raise DatasetNotFoundException(dataset_name=name)
            return self.db.collection(name=name)
        except CollectionListError as exc:
            raise DatasetNotFoundException(dataset_name=name) from exc

    def get_values(self, dataset_identifier: str, data_to_get: List[str]) -> None:
        """
        Method to retrieve data from JSON and fill a data_dict

        :param dataset_identifier: identifier of the dataset
        :type dataset_identifier: str

        :param data_to_get: data to retrieve, list of names
        :type data_to_get: List[str]
        """
        dataset_collection = self.__get_dataset_collection(name=dataset_identifier)

        # Retrieve the data
        cursor = dataset_collection.get_many(data_to_get)

        # Process the results
        result_data = {doc['_key']:doc['value'] for doc in cursor}
        return result_data


    def write_values(self, dataset_identifier: str, values_to_write: dict[str:Any]) -> None:
        """
        Method to write data

        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: List[str]
        """
        dataset_collection = self.__get_dataset_collection(name=dataset_identifier)
        # prepare query to write
        data_for_arango = [{"_key":tag, 'value': value} for tag, value in values_to_write.items()]

        # Write items
        dataset_collection.insert_many(data_for_arango, overwrite=True)

    def get_values_all(self, dataset_identifier: str) -> dict[str:Any]:
        """
        Get all values from a dataset for Arango
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        """
        dataset_collection = self.__get_dataset_collection(name=dataset_identifier)

        # Process all data
        result_data = {doc['_key']:doc['value'] for doc in dataset_collection}
        return result_data

    def write_dataset(self, dataset_identifier: str, values_to_write: dict[str:Any], create_if_not_exists:bool=True, override:bool=False) -> None:
        """
        Write a dataset from Arango
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str:Any]
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        """
        if not self.db.has_collection(name=dataset_identifier):
            # Handle dataset creation
            if create_if_not_exists:
                self.db.create_collection(name=dataset_identifier)
            else:
                raise DatasetNotFoundException(dataset_identifier)
        else:
            # Handle override
            if not override:
                raise DatasetGenericException(f"Dataset {dataset_identifier} would be overriden")
            
        self.write_values(dataset_identifier=dataset_identifier, values_to_write=values_to_write)
            

if __name__ == "__main__":
    """
    Example usage using docker deployment for arango
    """
    connector_values = {
        "host": 'http://127.0.0.1:8529',
        "db_name":'os-climate',
        "username":"root",
        "password":"ArangoDB_BfPM",
    }
    
    connector = ArangoDatasetsConnector(**connector_values)
    # Write values
    connector.write_values(dataset_identifier="test_dataset_collection", values_to_write={"x": 1, "y": "str_y2"})

    # Read values
    print(connector.get_values(dataset_identifier="test_dataset_collection", data_to_get=["x", "y"]))

    # Read dataset
    print(connector.get_all_values(dataset_identifier="test_dataset_collection"))

    # Write dataset
    connector.write_dataset(dataset_identifier="test_dataset_collection_2", values_to_write={"x": 1, "y": "str_y2"})
