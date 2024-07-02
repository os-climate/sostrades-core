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

import numpy as np
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
    
    SELF_TABLE_TYPES = ["dataframe", "dict"]
    NO_TABLE_TYPES = ["string", "int", "float", "bool", "array", "list"]

    DESCRIPTOR_TABLE_NAME = "descriptor_parameters"
    

    NAME = "parameter_name"
    STRING_VALUE = "parameter_string_value"
    INT_VALUE = "parameter_int_value"
    FLOAT_VALUE = "parameter_float_value"
    BOOL_VALUE = "parameter_bool_value"
    LIST_VALUE = "parameter_list_value"
    UNIT = "unit"
    DESCRIPTION = "description"
    LAST_UPDATE = "last_update_date"
    SOURCE = "source"
    LINK = "link"

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
        self.__datasets_serializer = DatasetsSerializerFactory.get_serializer(serializer_type)

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
        dataset_id = "{}.{}".format(self.client.project, dataset_identifier)

        # check dataset exists
        try:
            dataset = self.client.get_dataset(dataset_id)  # Make an API request.
            print("Dataset {} exists".format(dataset_id))
        except:
            raise DatasetNotFoundException(dataset_name=dataset_identifier)

        # Read dataset descriptor
        parameters_data = {}
        table_descriptor_id = "{}.{}".format(dataset_id, self.DESCRIPTOR_TABLE_NAME)
        try:
             table = self.client.get_table(table_descriptor_id)  # Make an API request.
        except:
            raise DatasetGenericException(f"Dataset {dataset_id}: {self.DESCRIPTOR_TABLE_NAME} table not found. Impossible to fetch parameter value.")
        
        json_descriptor = {}
        try:
            json_descriptor = self.__read_descriptor_table(table_descriptor_id)
        except Exception as ex:
            raise DatasetGenericException(f"Error while reading the Descriptor table for dataset {dataset_id}: {ex}") from ex
        
        
        for data, data_type in data_to_get.items():
            if data in json_descriptor.keys():
                if data_type in self.NO_TABLE_TYPES:
                    try:
                        parameters_data[data] = self.__datasets_serializer.convert_from_dataset_data(data, json_descriptor[data][self.__get_col_name_from_type(data_type)], {data:data_type})
                    except Exception as ex:
                        self.__logger.warning(f"Error while reading the parameter {data} in descriptor for dataset {dataset_id}: {ex}")
                elif data_type in self.SELF_TABLE_TYPES:
                    try:
                        # read and convert the data
                        if data_type == "dataframe" and json_descriptor[data][self.STRING_VALUE].startswith(f'@{data_type}@'):
                            table_id = "{}.{}".format(dataset_id, json_descriptor[data][self.STRING_VALUE].replace(f'@{data_type}@', ''))
                            parameters_data[data] =  self.__read_dataframe_table(table_id)
                        elif data_type == "dict" and json_descriptor[data][self.STRING_VALUE].startswith(f'@{data_type}@'):
                            table_id = "{}.{}".format(dataset_id, json_descriptor[data][self.STRING_VALUE].replace(f'@{data_type}@', ''))
                            parameters_data[data] =  self.__read_dict_table(table_id)
                    except Exception as ex:
                        self.__logger.warning(f"Error while reading the parameter {data} table for dataset {dataset_id}: {ex}")
            
            
        
        self.__logger.debug(
            f"Values obtained {list(parameters_data.keys())} for dataset {dataset_identifier} for connector {self}"
        )
        return parameters_data

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

        # first serialize into descriptor avec complex type parameters:
        json_descriptor, complex_parameters = self.__create_json_descriptor_and_parameters(values_to_write, data_types_dict)

        # write dataset descriptor table
        table_descriptor_id = "{}.{}".format(dataset_id, self.DESCRIPTOR_TABLE_NAME)
        table_descriptor_exists = True
        try:
             table = self.client.get_table(table_descriptor_id)  # Make an API request.
        except:
            table_descriptor_exists = False
            self.__logger.debug(f"create table for descriptor:{table_descriptor_id}")

        if table_descriptor_exists:
            try:
                # if the table exists, read it to write it again in full (best than request each param to see if it already exists)
                old_json_descriptor = self.__read_descriptor_table(table_descriptor_id)
                for data, metadata in old_json_descriptor.items():
                    if data in json_descriptor.keys():
                        # if the data was already in the dataset we don't overwrite the metadata
                        json_descriptor[data][self.UNIT] = metadata.get(self.UNIT, "")
                        json_descriptor[data][self.DESCRIPTION] = metadata.get(self.DESCRIPTION, "")
                        json_descriptor[data][self.SOURCE] = metadata.get(self.SOURCE, "")
                        json_descriptor[data][self.LINK] = metadata.get(self.LINK, "")
                        json_descriptor[data][self.LAST_UPDATE] = metadata.get(self.LAST_UPDATE, "")
                    else:
                        # the data already in the dataset is written again in it to not loose any information
                        json_descriptor[data] = metadata
            except Exception as ex:
                raise DatasetGenericException(f"Error while reading the Descriptor table for dataset {dataset_id}: {ex}") from ex
        
        try:
            # create or overwrite the descriptor table
            table = bigquery.Table(table_descriptor_id)
            job_config = bigquery.LoadJobConfig()
            job_config.autodetect = True
            job_config.write_disposition="WRITE_TRUNCATE"
            job = self.client.load_table_from_json(list(json_descriptor.values()), table, job_config = job_config)
            res = job.result()
        except Exception as ex:
            raise DatasetGenericException(f"Error while writing the Descriptor table for dataset {dataset_id}: {ex}") from ex

        #then write the complex parameters tables
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
                job_config.write_disposition="WRITE_TRUNCATE"

                try:
                    job = self.client.load_table_from_json([value], table, job_config = job_config)
                    res = job.result()
                    self.__logger.debug(f"created or updated table for data {data}:{res}")
                except Exception as ex:
                    raise DatasetGenericException(f"Error while writing the parameter data table {data}: {ex}") from ex

        return values_to_write

    def get_values_all(self, dataset_identifier: str, data_types_dict:dict[str:str]) -> dict[str:Any]:
        """
        Get all values from a dataset 
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        """
        
        return self.get_values(dataset_identifier, data_types_dict)
    
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

    def __create_json_descriptor_and_parameters(self, values_to_write: dict[str:Any], data_types_dict: dict[str:str])->tuple[dict, dict]:

        json_descriptor_parameters = {}
        complex_type_parameters_values = {}
        for parameter, parameter_value in values_to_write.items():
            parameter_type = data_types_dict[parameter]

            json_parameter = {
                self.NAME:parameter,
                self.UNIT:"",
                self.DESCRIPTION:"",
                self.SOURCE:"",
                self.LINK:""
            }

            #convert value in jsonifiable value:
            json_value = self.__datasets_serializer.convert_to_dataset_data(parameter, parameter_value, {parameter:parameter_type})

            # if data is none or empty (dict with no elements ect) bigquery raise an error at the import
            is_empty = json_value is None or (json_value is not None and 
                                              (parameter_type == "list" or parameter_type == "array" or parameter_type == "dict" or parameter_type == "dataframe") and
                                                len(json_value) == 0)
            if not is_empty:
                # simple parameters types have their value in the descriptor dict
                if parameter_type in BigqueryDatasetsConnector.NO_TABLE_TYPES:
                    column_name = self.__get_col_name_from_type(parameter_type)
                    json_parameter[column_name] = json_value
                    # there are some issues to export int64 or float 64 in json with big query so we need to convert it
                    if parameter_type == "int":
                        json_parameter[column_name] = int(json_parameter[column_name])
                    elif parameter_type == "float":
                        json_parameter[column_name] = float(json_parameter[column_name])
                    elif parameter_type == "list" or parameter_type == "array":
                        json_parameter[column_name] = self._serialize_sub_element_jsonifiable(json_parameter[column_name])
                elif parameter_type in BigqueryDatasetsConnector.SELF_TABLE_TYPES:
                    # the name of the value is a string value is the type of the data + the name of the table associated
                    json_parameter[self.STRING_VALUE] = f"@{parameter_type}@{parameter}"
                    complex_type_parameters_values[parameter] = json_value
                    
                
                # add the parameter row in json object
                json_descriptor_parameters[parameter] = json_parameter
        
        return json_descriptor_parameters, complex_type_parameters_values
    

    def __get_col_name_from_type(self, parameter_type:str)->str:
        '''
        retrieve column name from parameter type
        '''
        column_name = self.STRING_VALUE
        if parameter_type == "int":
            column_name = self.INT_VALUE
        elif parameter_type == "float":
            column_name = self.FLOAT_VALUE
        elif parameter_type == "bool":
            column_name = self.BOOL_VALUE
        elif parameter_type == "list" or parameter_type == "array":
            column_name = self.LIST_VALUE
        
        return column_name
    
    def __read_descriptor_table(self, descriptor_table_id:str)-> dict:
        '''
        read the descriptor parameter table
        :param descriptor_table_id: descriptor_parameter table id
        :type descriptor_table_id:str
        :return: dict[param_name: dict[metadata]]
        '''
        QUERY = (f'SELECT * FROM `{descriptor_table_id}` ')
        query_job = self.client.query(QUERY)  # API request
        return {row.parameter_name: {key:values for key, values in row.items()} for row in query_job.result()}
    
    def __read_dataframe_table(self, table_id:str)-> pd.DataFrame:
        '''
        read a dataframe parameter table
        :param table_id:  table id
        :type table_id:str
        :return: dataframe
        '''
        QUERY = (f'SELECT * FROM `{table_id}` ')
        return self.client.query(QUERY).result().to_dataframe()
    
    def __read_dict_table(self, table_id:str)-> dict:
        '''
        read a dict parameter table
        :param table_id:  table id
        :type table_id:str
        :return: dict
        '''
        QUERY = (f'SELECT * FROM `{table_id}` ')
        query_job = self.client.query(QUERY)
        result = [ {key:values for key, values in row.items()} for row in query_job.result()]
        if len(result) == 1:
            result = result[0]
        return result
    
    def _serialize_sub_element_jsonifiable(self, data_value:list):
        '''
        convert sub element of a list into non numpy format
        '''
        json_value = []
        for element in data_value:
            if isinstance(element, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):

               json_value.append(int(element))

            elif isinstance(element, (np.float_, np.float16, np.float32, np.float64)):
                json_value.append(float(element))
            else:
                json_value.append(element)

        return json_value