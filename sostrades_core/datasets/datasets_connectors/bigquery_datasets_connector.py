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

from sostrades_core.datasets.dataset_info.abstract_dataset_info import AbstractDatasetInfo
from sostrades_core.datasets.dataset_info.dataset_info_v0 import DatasetInfoV0
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetGenericException,
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
    SELF_TABLE_TYPES = ["dataframe", "dict", "array", "list"]
    NO_TABLE_TYPES = ["string", "int", "float", "bool"]
    DESCRIPTOR_TABLE_NAME = "descriptor_parameters"         # reserved table for dataset descriptor
    COL_NAME_INDEX_TABLE_NAME = "__col_name_index_table__"  # reserved table for bigquery characters compatibility
    URL_BQ_FORMAT = "https://console.cloud.google.com/bigquery?project={}&p={}&d={}&t={}&page=table"

    def __init__(self, connector_id: str, project_id: str,
                 serializer_type: DatasetSerializerType = DatasetSerializerType.BigQuery):
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
        self.__datasets_serializer = DatasetsSerializerFactory.get_serializer(serializer_type)
        self.connector_id = connector_id

        # Connect to database
        try:
            self.client = bigquery.Client(project=project_id)

        except Exception as exc:
            raise DatasetUnableToInitializeConnectorException(connector_type=BigqueryDatasetsConnector) from exc

    def _get_values(self, dataset_identifier: AbstractDatasetInfo, data_to_get: dict[str, str]) -> dict[str, Any]:
        """
        Method to retrieve data from dataset and fill a data_dict

        Args:
            dataset_identifier (AbstractDatasetInfo): identifier of the dataset
            data_to_get (dict[str, str]): data to retrieve, dict of names and types

        Returns:
            dict[str, Any]: dict of retrieved data
        """

        dataset_id = "{}.{}".format(self.client.project, dataset_identifier.dataset_id)
        json_descriptor, col_name_index, _, _ = self.__load_descriptor_index_tables(dataset_id)
        parameters_data = dict()
        self.__datasets_serializer.set_col_name_index(col_name_index)
        for data, data_type in data_to_get.items():
            if data in json_descriptor.keys():
                if data_type in self.NO_TABLE_TYPES:
                    try:
                        parameters_data[data] = self.__datasets_serializer.convert_from_dataset_data(data, json_descriptor[data][self.__get_col_name_from_type(data_type)], {data: data_type})
                    except Exception as ex:
                        self.__logger.warning(f"Error while reading the parameter {data} in descriptor for dataset {dataset_id}: {ex}")
                elif data_type in self.SELF_TABLE_TYPES:
                    try:
                        # read and convert the data
                        if json_descriptor[data][self.STRING_VALUE].startswith(f'@{data_type}@'):
                            table_id = "{}.{}".format(
                                dataset_id, json_descriptor[data][self.STRING_VALUE].replace(f'@{data_type}@', ''))
                            if data_type == "dataframe":
                                parameters_data[data] = self.__datasets_serializer.convert_from_dataset_data(
                                    data, self.__read_dataframe_table(table_id), {data: data_type})
                            else:
                                parameters_data[data] = self.__datasets_serializer.convert_from_dataset_data(
                                    data, self.__read_dict_table(table_id), {data: data_type})
                    except Exception as ex:
                        self.__logger.warning(f"Error while reading the parameter {data} table for dataset {dataset_id}: {ex}")

        self.__datasets_serializer.clear_col_name_index()
        self.__logger.debug(
            f"Values obtained {list(parameters_data.keys())} for dataset {dataset_identifier.dataset_id} for connector {self}"
        )
        return parameters_data

    def _write_values(self, dataset_identifier: AbstractDatasetInfo, values_to_write: dict[str, Any], data_types_dict: dict[str, str]) -> dict[str, Any]:
        """
        Method to write data

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            values_to_write (dict[str, Any]): dict of data to write {name: value}
            data_types_dict (dict[str, str]): dict of data type {name: type}

        Returns:
            dict[str, Any]: dict of written values
        """
        self.__logger.debug(f"Writing values in dataset {dataset_identifier.dataset_id} for connector {self}")

        dataset_id = "{}.{}".format(self.client.project, dataset_identifier.dataset_id)
        old_json_descriptor, old_index, table_descriptor_id, table_index_id = self.__load_descriptor_index_tables(dataset_id=dataset_id)
        # serialize into descriptor with complex type parameters:
        self.__datasets_serializer.set_col_name_index(old_index)
        json_descriptor, complex_parameters = self.__update_json_descriptor_and_parameters(values_to_write,
                                                                                           old_json_descriptor,
                                                                                           data_types_dict)
        new_index = self.__datasets_serializer.col_name_index
        self.__datasets_serializer.clear_col_name_index()

        try:
            # create or overwrite the descriptor table
            table = bigquery.Table(table_descriptor_id)
            job_config = bigquery.LoadJobConfig()
            job_config.autodetect = True
            job_config.write_disposition = "WRITE_TRUNCATE"
            job = self.client.load_table_from_json(list(json_descriptor.values()), table, job_config=job_config)
            res = job.result()
        except Exception as ex:
            raise DatasetGenericException(f"Error while writing the Descriptor table for dataset {dataset_id}: {ex}") from ex

        # then write the complex parameters tables
        for data, value in complex_parameters.items():
            # check that the value we want to write in a table has data,
            # if not bigquery api will raise an error
            if len(value) > 0:
                # get table in dataset
                table_id = "{}.{}".format(dataset_id, data)
                table = bigquery.Table(table_id)

                job_config = bigquery.LoadJobConfig()
                job_config.autodetect = True
                # the data is overwrited
                job_config.write_disposition = "WRITE_TRUNCATE"

                try:
                    if data_types_dict[data] == "dataframe":
                        job = self.client.load_table_from_dataframe(value, table, job_config=job_config)
                    else:
                        job = self.client.load_table_from_json([value], table, job_config=job_config)

                    res = job.result()
                    self.__logger.debug(f"created or updated table for data {data}:{res}")
                except Exception as ex:
                    raise DatasetGenericException(f"Error while writing the parameter data table {data}: {ex}") from ex

        # then update the dataframe and dict col name index
        try:
            # create or overwrite the descriptor table
            table = bigquery.Table(table_index_id)
            job_config = bigquery.LoadJobConfig()
            job_config.autodetect = True
            job_config.write_disposition = "WRITE_TRUNCATE"
            job = self.client.load_table_from_json([new_index], table, job_config=job_config)
            res = job.result()
        except Exception as ex:
            raise DatasetGenericException(f"Error while writing the Descriptor table for dataset {dataset_id}: {ex}") from ex

        return values_to_write

    def _get_values_all(self, dataset_identifier: AbstractDatasetInfo, data_types_dict: dict[str, str]) -> dict[str, Any]:
        """
        Get all values from a dataset

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            data_types_dict (dict[str, str]): dict of data type {name: type}

        Returns:
            dict[str, Any]: dict of all values
        """

        return self.get_values(dataset_identifier, data_types_dict)

    def get_datasets_available(self) -> list[AbstractDatasetInfo]:
        """
        Get all available datasets for a specific API

        Returns:
            list[AbstractDatasetInfo]: list of available datasets
        """
        self.__logger.debug(f"Getting all datasets for connector {self}")
        datasets = list(self.client.list_datasets())
        dataset_identifiers = [DatasetInfoV0(self.connector_id, dataset_id) for dataset_id in datasets]
        return dataset_identifiers

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
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            values_to_write (dict[str, Any]): dict of data to write {name: value}
            data_types_dict (dict[str, str]): dict of data types {name: type}
            create_if_not_exists (bool, optional): create the dataset if it does not exists (raises otherwise). Defaults to True.
            override (bool, optional): override dataset if it exists (raises otherwise). Defaults to False.

        Returns:
            dict[str, Any]: dict of written values
        """
        self.__logger.debug(
            f"Writing dataset {dataset_identifier.dataset_id} for connector {self} (override={override}, create_if_not_exists={create_if_not_exists})"
        )
        dataset_id = "{}.{}".format(self.client.project, dataset_identifier.dataset_id)
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

    def _build_path_to_data(self, dataset_identifier: AbstractDatasetInfo, data_name: str, data_type: str) -> str:
        """
        Overloaded method in order to build the url to the bigquery data

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier into connector
            data_name (str): data in dataset
            data_type (str): type of the data in dataset

        Returns:
            str: path/url/uri to find the dataset data
        """
        path_to_data = ""

        # if data is simple data, the link is to the descriptor table
        if data_type in ["int", "float", "bool", "string"]:
            table_id = self.DESCRIPTOR_TABLE_NAME
        else:
            # if data is a dataframe or other, the data has its own table
            table_id = data_name
        path_to_data = self.URL_BQ_FORMAT.format(self.client.project,
                                                 self.client.project,
                                                 dataset_identifier.dataset_id,
                                                 table_id)

        return path_to_data

    def _insert_value_into_datum(self,
                                 value: Any,
                                 datum: dict[str, Any],
                                 parameter_name: str,
                                 parameter_type: str) -> dict[str, Any]:

        new_datum = self._new_datum(datum)
        new_datum[self.PARAMETER_NAME] = parameter_name
        # if data is none or empty (dict with no elements ect) bigquery raise an error at the import
        is_empty = value is None or (parameter_type in {"list", "array", "dict", "dataframe"} and len(value) == 0)
        if not is_empty:
            # simple parameters types have their value in the descriptor dict
            if parameter_type in BigqueryDatasetsConnector.NO_TABLE_TYPES:
                column_name = self.__get_col_name_from_type(parameter_type)
                # there are some issues to export int64 or float 64 in json with big query so we need to convert it
                if parameter_type == "int":
                    new_datum[column_name] = int(value)
                elif parameter_type == "float":
                    new_datum[column_name] = float(value)
                else:
                    new_datum[column_name] = value
            elif parameter_type in BigqueryDatasetsConnector.SELF_TABLE_TYPES:
                # the name of the value is a string value is the type of the data + the name of the table associated
                new_datum[self.STRING_VALUE] = f"@{parameter_type}@{parameter_name}"
            # TODO: keeping dataset metadata "as is", insert metadata handling here
            return new_datum

    def __update_json_descriptor_and_parameters(self,
                                                values_to_write: dict[str, Any],
                                                old_json_descriptor: dict[str, dict[str, Any]],
                                                data_types_dict: dict[str, str]) -> tuple[dict, dict]:
        """
        Update the JSON descriptor and parameters

        Args:
            values_to_write (dict[str, Any]): values to write
            old_json_descriptor (dict[str, dict[str, Any]]): old JSON descriptor
            data_types_dict (dict[str, str]): data types dictionary

        Returns:
            tuple[dict, dict]: updated JSON descriptor and complex type parameters values
        """
        json_descriptor_parameters = old_json_descriptor or {}
        json_values = {parameter: self.__datasets_serializer.convert_to_dataset_data(parameter, parameter_value,
                                                                                     {parameter: data_types_dict[parameter]})
                       for parameter, parameter_value in values_to_write.items()}
        # update the descriptor
        self._update_data_with_values(json_descriptor_parameters, json_values, data_types_dict)

        # recover the complex types values
        complex_type_parameters_values = {parameter: json_values[parameter] for parameter in json_values
                                          if parameter in json_descriptor_parameters and
                                          json_descriptor_parameters[parameter].get(self.STRING_VALUE, "") ==
                                          f"@{data_types_dict[parameter]}@{parameter}"}

        return json_descriptor_parameters, complex_type_parameters_values

    def __get_col_name_from_type(self, parameter_type: str) -> str:
        """
        Retrieve column name from parameter type

        Args:
            parameter_type (str): type of the parameter

        Returns:
            str: column name
        """
        column_name = self.STRING_VALUE
        if parameter_type == "int":
            column_name = self.INT_VALUE
        elif parameter_type == "float":
            column_name = self.FLOAT_VALUE
        elif parameter_type == "bool":
            column_name = self.BOOL_VALUE

        return column_name

    def __load_descriptor_index_tables(self, dataset_id: str) -> tuple[dict, dict, str, str]:
        """
        Load descriptor and index tables

        Args:
            dataset_id (str): id of the dataset

        Returns:
            tuple[dict, dict, str, str]: tuple of descriptor, index, descriptor table id, and index table id
        """
        # load dataset descriptor table
        table_descriptor_id = "{}.{}".format(dataset_id, self.DESCRIPTOR_TABLE_NAME)
        table_descriptor_exists = True
        try:
             table = self.client.get_table(table_descriptor_id)  # Make an API request.
        except:
            table_descriptor_exists = False
            self.__logger.debug(f"create table for descriptor:{table_descriptor_id}")
        json_descriptor_read = dict()
        if table_descriptor_exists:
            try:
                # if the table exists, read it to write it again in full (best than request each param to see if it already exists)
                json_descriptor_read = self.__read_descriptor_or_index_table(table_descriptor_id)
            except Exception as ex:
                raise DatasetGenericException(f"Error while reading the Descriptor table for dataset {dataset_id}: {ex}") from ex

        # load dataset column name index table
        table_index_id = "{}.{}".format(dataset_id, self.COL_NAME_INDEX_TABLE_NAME)
        table_index_exists = True
        try:
             table = self.client.get_table(table_index_id)  # Make an API request.
        except:
            table_index_exists = False
            self.__logger.debug(f"create table for descriptor:{table_index_id}")
        index_read = dict()
        if table_index_exists:
            try:
                # if the table exists, read it to write it again in full (best than request each param to see if it already exists)
                index_read = self.__read_dict_table(table_index_id)
            except Exception as ex:
                self.__logger.debug(f"Error while reading the {self.COL_NAME_INDEX_TABLE_NAME} for dataset {dataset_id}:"
                                    f" {ex}, assuming an empty index.")
        return json_descriptor_read, index_read, table_descriptor_id, table_index_id

    def __read_descriptor_or_index_table(self, table_id: str) -> dict:
        """
        Read the descriptor parameter table

        Args:
            table_id (str): descriptor_parameter table id

        Returns:
            dict: dict of parameters and their metadata
        """
        QUERY = (f'SELECT * FROM `{table_id}` ')
        query_job = self.client.query(QUERY)  # API request
        return {row.parameter_name: {key: values for key, values in row.items()} for row in query_job.result()}

    def __read_dataframe_table(self, table_id: str) -> pd.DataFrame:
        """
        Read a dataframe parameter table

        Args:
            table_id (str): table id

        Returns:
            pd.DataFrame: dataframe
        """
        QUERY = (f'SELECT * FROM `{table_id}` ')
        return self.client.query(QUERY).result().to_dataframe()

    def __read_dict_table(self, table_id: str) -> dict:
        """
        Read a dict parameter table

        Args:
            table_id (str): table id

        Returns:
            dict: dict
        """
        QUERY = (f'SELECT * FROM `{table_id}` ')
        query_job = self.client.query(QUERY)
        result = [{key: values for key, values in row.items()} for row in query_job.result()]
        if len(result) == 1:
            result = result[0]
        return result
