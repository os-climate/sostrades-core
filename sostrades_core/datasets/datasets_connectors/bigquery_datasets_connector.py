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
from typing import Any
import pandas as pd
from google.cloud import bigquery


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


class BigqueryDatasetsConnector(AbstractDatasetsConnector):
    """
    Specific dataset connector for dataset in google cloud bigquery db format
    """
    
    def __init__(self, project_id: str,
                 serializer_type:DatasetSerializerType=DatasetSerializerType.JSON):
        """
        Constructor for Arango data connector

        :param project_id: id of the gcp
        :type project_id: str

        :param serializer_type: type of serializer to deserialize data from connector
        :type serializer_type: DatasetSerializerType (JSON for jsonDatasetSerializer)

        """
        super().__init__()
        self.__logger = logging.getLogger(__name__)
        self.__logger.debug("Initializing Bigquery connector")
        #self.__datasets_serializer = DatasetsSerializerFactory.get_serializer(serializer_type)

        # Connect to database
        try:
            self.client = bigquery.Client(project=project_id)

        except Exception as exc:
            raise DatasetUnableToInitializeConnectorException(connector_type=BigqueryDatasetsConnector) from exc

   
    def get_values(self, dataset_identifier: str, data_to_get: dict[str:str]) -> None:
        """
        Method to retrieve data from dataset and fill a data_dict

        :param dataset_identifier: identifier of the dataset
        :type dataset_identifier: str

        :param data_to_get: data to retrieve, dict of names and types
        :type data_to_get: dict[str:str]
        """
        self.__logger.debug(f"Getting values {data_to_get.keys()} for dataset {dataset_identifier} for connector {self}")
        
        dataset_id = "{}.{}".format(self.client.project, dataset_identifier)

        # check dataset exists
        try:
            dataset = self.client.get_dataset(dataset_id)  # Make an API request.
            print("Dataset {} exists".format(dataset_id))
        except:
            raise DatasetNotFoundException(dataset_name=dataset_identifier)

        # Read dataset dataframe
        result_data = {}
        for data, type in data_to_get.items():
            if type == "dataframe":
                # get table in dataset
                table_id = "{}.{}".format(dataset_id, data)
                # check table exists
                try:
                    self.client.get_table(table_id)
                    QUERY = (
                    f'SELECT * FROM `{table_id}`')
                    query_job = self.client.query(QUERY)  # API request
                    result_data[data] = query_job.result().to_dataframe()
                    self.__logger.info(f"Value of {data}:{result_data[data]} retrieved in dataset {dataset_identifier} for connector {self}")
                    
                except :
                    self.__logger.debug(f"Value of {data} is not in dataset {dataset_identifier} for connector {self}")

        self.__logger.debug(f"Values obtained {list(result_data.keys())} for dataset {dataset_identifier} for connector {self}"
        )
        return result_data

    def write_values(self, dataset_identifier: str, values_to_write: dict[str:Any], data_types_dict: dict[str:str]) -> dict[str: Any]:
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
        dataset_id = "{}.{}".format(self.client.project, dataset_identifier)

        # write dataset table
        for data, value in values_to_write.items():
            if data_types_dict[data] == "dataframe":
                # get table in dataset
                table_id = "{}.{}".format(dataset_id, data)
                table = bigquery.Table(table_id)

                job_config = bigquery.LoadJobConfig()
                #job_config.schema = schema
                job_config.autodetect = True
                job_config.write_disposition="WRITE_TRUNCATE"

                # need requirement pyarrow
                job = self.client.load_table_from_dataframe(value, table, job_config = job_config)
                res = job.result()
                self.__logger.debug("create table for data {data}:{res}")
                # exit to write only one value (for testing purpose and not use too much storage)
                break

        return values_to_write

    def get_values_all(self, dataset_identifier: str, data_types_dict:dict[str:str]) -> dict[str:Any]:
        """
        Get all values from a dataset for Arango
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        """
        dataset_id = "{}.{}".format(self.client.project, dataset_identifier)

        # check dataset exists
        try:
            dataset = self.client.get_dataset(dataset_id)  # Make an API request.
            print("Dataset {} exists".format(dataset_id))
        except:
            raise DatasetNotFoundException(dataset_name=dataset_identifier)

        # Read dataset dataframe
        result_data = {}
        for data, type in data_types_dict.items():
            if type == "dataframe":
                # get table in dataset
                table_id = "{}.{}".format(dataset_id, data)
                QUERY = (
                f'SELECT * FROM `{table_id}`')
                query_job = self.client.query(QUERY)  # API request
                result_data[data] = query_job.result().to_dataframe()
        
        self.__logger.debug(
            f"Values obtained {list(result_data.keys())} for dataset {dataset_identifier} for connector {self}"
        )
        return result_data
    
    def get_datasets_available(self) -> list[str]:
        """
        Get all available datasets for a specific API
        """
        self.__logger.debug(f"Getting all datasets for connector {self}")
        datasets = list(self.client.list_datasets())
        return datasets

    def write_dataset(
        self,
        dataset_identifier: str,
        values_to_write: dict[str:Any],
        data_types_dict:dict[str:str],
        create_if_not_exists: bool = True,
        override: bool = False,
    ) -> dict[str: Any]:
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
        dataset_id = "{}.{}".format(self.client.project, dataset_identifier)
        # check dataset exists
        try:
            dataset = self.client.get_dataset(dataset_id)  # Make an API request.
            self.__logger.debug("Dataset {} exists".format(dataset_id))
        except:
            if create_if_not_exists:
                # if dataset doesn't exists, create the dataset
                # Construct a full Dataset object to send to the API.
                dataset = bigquery.Dataset(dataset_id)

                # TODO: Specify the geographic location where the dataset should reside.
                dataset.location = "europe-west1"

                # Send the dataset to the API for creation, with an explicit timeout.
                # Raises google.api_core.exceptions.Conflict if the Dataset already
                # exists within the project.
                dataset = self.client.create_dataset(dataset, timeout=30) 



        return self.write_values(dataset_identifier=dataset_identifier, values_to_write=values_to_write, data_types_dict=data_types_dict)
