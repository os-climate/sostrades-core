# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
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
import trino
import re
import copy


class TrinoDataConnector(AbstractDataConnector):
    """
    Specific data connector for Trino
    """

    data_connection_list = ['hostname',
                            'port', 'username', 'catalog', 'schema']

    NAME = 'TRINO'

    CONNECTOR_TYPE = 'connector_type'
    CONNECTOR_DATA = 'connector_data'
    CONNECTOR_TABLE = 'connector_table'
    CONNECTOR_CONDITION = 'connector_condition'

    COLUMN_NAME = 0
    COLUMN_TYPE = 1
    COLUMN_UNKNOWN_1 = 2
    COLUMN_UNKNOWN_2 = 3
    COLUMN_REGEXP = "^row\\((.*)\\)$"

    table_columns_definition = {}

    def __init__(self, data_connection_info=None):
        """
        Constructor for Dremio data connector

        :param data_connection_info: contains necessary data for connection
        :type data_connection_info: dict
        """

        self.hostname = None
        self.port = None
        self.username = None
        self.catalog = None
        self.schema = None

        super().__init__(data_connection_info=data_connection_info)

    def _extract_connection_info(self, data_connection_info):
        """
        Convert structure with data connection info given as parameter into member variable

        :param data_connection_info: contains necessary data for connection
        :type data_connection_info: dict
        """

        self.hostname = data_connection_info['hostname']
        self.port = data_connection_info['port']
        self.username = data_connection_info['username']
        self.catalog = data_connection_info['catalog']
        self.schema = data_connection_info['schema']

    def load_data(self, connection_data):
        """
        Method to load a data from Trino regarding input information

        :param: connection_data_dict, contains the necessary information to connect to Trino API with request
        :type: dict

        """

        self._extract_connection_info(connection_data)

        # Connect to Trino api
        trino_connection = trino.dbapi.connect(
            host=self.hostname,
            port=self.port,
            user=self.username,
            catalog=self.catalog,
            schema=self.schema,
            http_scheme='http')

        # Get the data from dremio
        table = connection_data[self.CONNECTOR_TABLE]
        condition = connection_data[self.CONNECTOR_CONDITION]
        sql = f'SELECT * FROM {table}'

        if condition:
            sql = f'{sql} WHERE {condition}'

        connection_cursor = trino_connection.cursor()
        connection_cursor.execute(sql)
        rows = connection_cursor.fetchall()

        self.__update_table_column(connection_cursor, table)

        return self.__map_data_with_table_column(rows, table)

    def write_data(self, connection_data):

        raise Exception("method not implemented")

    def set_connector_request(self, connector_info, table, condition):
        """
        Update connector dictionary with request information
        :param connector_info: dictionary regarding connection information, must map TrinoDataConnector.data_connection_list
        :param table: target table name for the request
        :param condition: condition to implement in SQL format
        :return:
        """

        connector_info[TrinoDataConnector.CONNECTOR_TABLE] = table
        connector_info[TrinoDataConnector.CONNECTOR_CONDITION] = condition
        return connector_info

    def __update_table_column(self, connection_cursor, table):
        """
        Send a request to Trino in order to get the column name of the table use for the request.
        Columns definition will be stored in a static dictionary in order to improve subsequent treatment

        :param connection_cursor: current connection cursor to use
        :type connection_cursor: str

        :param table: table to get the definition
        :type table: str

        :return: column list
        """

        if table not in TrinoDataConnector.table_columns_definition:
            connection_cursor.execute(f'SHOW COLUMNS FROM {table}')
            rows = connection_cursor.fetchall()
            TrinoDataConnector.table_columns_definition[table] = self.__get_column_from_rows(rows)

    def __get_column_from_rows(self, request_rows):
        """
        translate request result from Trino into a dictionary structure with table columns as key

        :param request_rows:  Trino show columns for table request
        :return: organize dictionary with column structure
        """

        columns_definition = {}
        for row in request_rows:
            column_name = row[TrinoDataConnector.COLUMN_NAME]
            column_type = row[TrinoDataConnector.COLUMN_TYPE]

            sub_object = re.findall(TrinoDataConnector.COLUMN_REGEXP, column_type)

            if len(sub_object) > 0:
                # Split sub string which in the form
                # 'name1 type_name1, name2 type_name2, ....
                split_by_comma = sub_object[0].split(',')

                for index in range(len(split_by_comma)):
                    split_by_comma[index] = split_by_comma[index].strip().split(' ')

                sub_definition = self.__get_column_from_rows(split_by_comma)
                columns_definition[column_name] = sub_definition
            else:
                columns_definition[column_name] = column_type

        return columns_definition

    def __map_data_with_table_column(self, request_rows, table_name):
        """
        Using column definition build dictionary  that map attribute name with their values
        :param request_rows: Trino request result
        :param table_name: corresponding table
        :return: dictionary list that map attribute with their value
        """

        results = []

        for one_result in request_rows:
            # Get table definition
            table_definition = copy.deepcopy(TrinoDataConnector.table_columns_definition[table_name])
            self.__insert_list_value(table_definition, one_result)
            results.append(table_definition)

        return results

    def __insert_list_value(self, dictionary_to_update, list_to_insert):
        """
        Manage mapping for a value list regarding dictionary key
        :param dictionary_to_update: Current dictionary to update
        :param list_to_insert: Datalist to insert
        """

        values_index = 0

        for key in dictionary_to_update.keys():
            value = list_to_insert[values_index]
            values_index += 1
            if isinstance(value, list):
                self.__insert_list_value(dictionary_to_update[key], value)
            else:
                dictionary_to_update[key] = value


if __name__ == '__main__':

    trino_connector = TrinoDataConnector()

    data_connection = {
                        'hostname': 'idlvsrv201.eu.airbus.corp',
                        'port': 30300,
                        'username': 'sostrades',
                        'catalog': 'mongodb',
                        'schema': 'world'
                       }
    trino_connector.set_connector_request(data_connection, 'TECHNO', None)

    result = trino_connector.load_data(data_connection)

    print(result)


