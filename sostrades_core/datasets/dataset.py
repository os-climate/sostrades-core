from dataclasses import dataclass

from sostrades_core.datasets.dataset_info import DatasetInfo
from sostrades_core.execution_engine.data_connector.abstract_data_connector import AbstractDataConnector


@dataclass()
class Dataset:
    dataset_info:DatasetInfo
    connector:AbstractDataConnector

    def fetch_values(self, data_names):
        """
        Fetch dataset data and return a data dict with values
        """
        fetched_data = {}
        dataset_dict = self.connector.load_data(self.dataset_info.dataset_id)
        for data_name in data_names:
            if data_name in dataset_dict:
                fetched_data[data_name] = dataset_dict[data_name]
        
        return fetched_data
