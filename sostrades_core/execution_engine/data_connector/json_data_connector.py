'''
Copyright 2022 Airbus SAS

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

from sostrades_core.execution_engine.data_connector.abstract_data_connector import AbstractDataConnector
import json 
import pandas as pd 


def convert_from_editable_json(json_str):
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if len(obj) > 0 and isinstance(obj[0], dict):
                df = pd.DataFrame({k: v for d in obj for k, v in d.items()})
                #df = df.apply(lambda x: pd.to_numeric(x, errors='ignore') if x.dtype == np.object else x)
                return df
            else:
                return [convert(elem) for elem in obj]
        elif isinstance(obj, str):
            try:
                return int(obj)
            except ValueError:
                pass
            try:
                return float(obj)
            except ValueError:
                pass
            return obj
        else:
            return obj

    data = json.loads(json_str)
    return convert(data)


class JSONDataConnector(AbstractDataConnector):
    """
    Specific data connector for Dremio
    """

    

    NAME = 'JSON'

    CONNECTOR_TYPE = 'connector_type'
    CONNECTOR_DATA = 'connector_data'
    CONNECTOR_REQUEST = 'connector_request'

    def __init__(self, data_connection_info=None):
        """
        Constructor for JSON data connector

        :param data_connection_info: contains necessary data for connection
        :type data_connection_info: dict
        """



        super().__init__(data_connection_info=data_connection_info)

    def _extract_connection_info(self):
        """

        """
        raise Exception("method not implemented")


    def load_data(self, connection_data, database_name):
        """
        Method to load a data from JSON

        :param: connection_data_dict, contains the necessary information to connect to Dremio API : URL, port, usename, password
        :type: dict

        """


        # Read JSON
        with open(connection_data, "r") as f:
            json_data = f.read()

        data_dict_json = convert_from_editable_json(json_data)
        
        return data_dict_json[database_name]

    def write_data(self, connection_data):
        """
        Method to load a data from Dremio

        :param: connection_data_dict, contains the necessary information to connect to Dremio API : URL, port, usename, password
        :type: dict

        :param: dremio_path, identification of the data in dremio
        :type: string
        """

        raise Exception("method not implemented")

    def set_connector_request(self, connector_info, request):

        raise Exception("method not implemented")
