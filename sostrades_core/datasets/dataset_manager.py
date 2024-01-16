from sostrades_core.datasets.dataset_info import DatasetInfo
from sostrades_core.datasets.dataset import Dataset
from sostrades_core.datasets.datasets_connectors.json_datasets_connector import JSONDatasetsConnector


class DatasetManager:
    def __init__(self):
        self.datasets = {}

    def fetch_data_from_dataset(self, datasets_info, data_names):
        '''
        get data from datasets and fill data_dict

        :param: datasets_info, list of datasets associated to a namespace
        :type: list of DatasetInfo

        :param: data_names, list of data to be fetch in datasets
        :type: list of string (data names)

        :return: data_dict of data names and retrieved values
        '''
        updated_data = {}

        for dataset_info in datasets_info:
            # create a Dataset and save it in a list if it does not already exists
            if dataset_info not in self.datasets:
                self.datasets[dataset_info] = self._create_dataset(dataset_info)
            
            # retrieve the data values from the dataset
            updated_data.update(self.datasets[dataset_info].fetch_values(data_names))

        return updated_data

    def _create_dataset(self, dataset_info):
        '''Get the connector associated to the dataset and create a Dataset object
        '''

        #TODO: this connector should be saved in a connector factory singleton somewhere
        # and here we should only get the connector
        connector_connexion_info ={'filename': 'datasets_db.json'}
        connector = JSONDatasetsConnector(data_connection_info=connector_connexion_info)
             
        return Dataset(dataset_info=dataset_info, connector=connector)


     