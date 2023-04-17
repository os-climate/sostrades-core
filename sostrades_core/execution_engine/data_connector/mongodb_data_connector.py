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
import pandas as pd 
import os
from pymongo import MongoClient
import logging
import urllib.parse

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
    return convert(json_str)

def postprocess_json(data):
    """
    Replaces all occurrences of # in the keys of a dictionary with .

    Args:
        data (dict): The dictionary to be processed.

    Returns:
        dict: The processed dictionary.
    """
    processed_data = {}
    for key, value in data.items():
        processed_key = key.replace('#', '.')
        if isinstance(value, dict):
            processed_value = postprocess_json(value)
        elif isinstance(value, list):
            processed_value = value
        else:
            processed_value = value
        processed_data[processed_key] = processed_value
    return processed_data



def get_document_from_cosmosdb_pymongo(connection_string: str, database_name: str, collection_name: str, query: dict):
    """
    Connects to a MongoDB database using a connection string, and retrieves a single document from a collection
    using a query.

    :param connection_string: a MongoDB connection string
    :param database_name: the name of the database
    :param collection_name: the name of the collection
    :param query: a dictionary representing the query to use to find the document
    :return: a dictionary representing the retrieved document, or None if no document is found
    """

    # Create a MongoClient object using the connection string
    client = MongoClient(connection_string)

    # Access the specified database and collection
    db = client[database_name]
    collection = db[collection_name]

    # Use the find_one method to retrieve a single document matching the query
    document = collection.find_one(query)

    # Close the client connection
    client.close()
    logging.info(f'MongoDB is used, query {query} is loaded on database {database_name} for collection {collection_name}')
    # Return the retrieved document, or None if no document is found
    return document

class MongoDBDataConnector(AbstractDataConnector):
    """
    Specific data connector for MongoDB
    """

    NAME = 'MongoDB'

    CONNECTOR_TYPE = 'connector_type'
    CONNECTOR_DATA = 'connector_data'
    CONNECTOR_REQUEST = 'connector_request'

    def __init__(self):
        """
        Constructor for JSON data connector
        :param data_connection_info: contains necessary data for connection
        :type data_connection_info: dict
        """

        super().__init__()

    def _extract_connection_info(self):
        """

        """
        # NotImplementedError
        raise Exception("method not implemented")


    def load_data(self, database_id):
        """
        Method to load data from a specified database in Azure Cosmos DB.
        The method retrieves the connection string, database name, and collection name from environment variables,
        and queries for the database with the specified id using the get_document_from_cosmosdb_pymongo function.
        The returned document is then preprocessed and converted from editable JSON using the preprocess_json and
        convert_from_editable_json functions.
        The final preprocessed document is returned.
        :param database_id: The ID of the database to load.
        :return: The preprocessed database document.
        """

        connection_string = os.environ.get('COSMOSDB_CONNECTION')
        print(connection_string)
        connection_string_unquote = urllib.parse.unquote(connection_string)
        database_name = os.environ.get('COSMOSDB_NAME')
        collection_name = os.environ.get('COSMOSDB_COLLECTION')
        database = get_document_from_cosmosdb_pymongo(connection_string= connection_string_unquote, database_name = database_name, collection_name = collection_name, query = {'id': database_id})
        database_postproc = convert_from_editable_json(postprocess_json(database))
        return database_postproc 
    
    def write_data(self):
        """
        Method to load a data 
        """

        raise Exception("method not implemented")

    def set_connector_request(self):

        raise Exception("method not implemented")
