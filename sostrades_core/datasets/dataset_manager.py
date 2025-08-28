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
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from sostrades_core.datasets.dataset import Dataset
from sostrades_core.datasets.dataset_info.abstract_dataset_info import AbstractDatasetInfo
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    DatasetGenericException,
)
from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import (
    DatasetsConnectorManager,
)
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline

if TYPE_CHECKING:
    import logging


class DatasetsManager:
    """Manages connections to datasets"""

    VALUE = ProxyDiscipline.VALUE
    DATASET_INFO = 'dataset_info'

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initializes the DatasetsManager.

        Args:
            logger (logging.Logger): Logger instance for logging.

        """
        self.datasets = {}
        self.__logger = logger

    def fetch_data_from_datasets(self, datasets_info: Dict[AbstractDatasetInfo, Dict[str, str]],
                                 data_dict: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetches data from datasets and fills the data_dict.

        Args:
            datasets_info (Dict[AbstractDatasetInfo, Dict[str, str]]): List of datasets associated to a namespace.
            data_dict (Dict[str, str]): Dict of data to be fetched in datasets with their types.

        Returns:
            Dict[str, Dict[str, Any]]: Data dict of data names and retrieved values plus a DATASET_INFO field with DatasetInfo object.

        """
        self.__logger.debug(f"Fetching data {data_dict.keys()} from datasets {datasets_info}")
        data_retrieved = {}

        for dataset_info, mapping_parameters in datasets_info.items():
            try:
                # Get the dataset, creates it if not exists
                dataset = self.get_dataset(dataset_info=dataset_info)

                # get the list of parameters to get
                data_to_fetch = {}
                dataset_data_reverse_mapping = {}
                # we get the data_dataset_key for each param that is in data_dict
                # it is done in a loop so that it respect the order of appearance
                # (ie: if there is a *:* and then a:b, the a:b replace the *:* for the 'a' parameter)
                for data_key, data_dataset_key in mapping_parameters.items():
                    if data_dataset_key == AbstractDatasetInfo.WILDCARD:
                        dataset_data_reverse_mapping.update({key: key for key in data_dict.keys()})
                        data_to_fetch.update(data_dict)
                    elif data_key in data_dict.keys():
                        dataset_data_reverse_mapping.update({data_dataset_key: data_key})
                        data_to_fetch.update({data_dataset_key: data_dict[data_key]})

                # Retrieve values
                dataset_values = dataset.get_values(data_dict=data_to_fetch)
                # Update internal dictionary adding provenance (DatasetInfo object) for tracking parameter changes
                dataset_data = {dataset_data_reverse_mapping[key]: {self.VALUE: value,
                                    self.DATASET_INFO: dataset_info} for key, value in dataset_values.items()}
                data_retrieved.update(dataset_data)
            except DatasetGenericException as exception:
                raise DatasetGenericException(f'Error fetching dataset "{dataset_info.dataset_id}" of datasets connector "{dataset_info.connector_id}": {exception}')
        return data_retrieved

    def get_dataset(self, dataset_info: AbstractDatasetInfo) -> Dataset:
        """
        Gets a dataset, creates it if it does not exist.

        Args:
            dataset_info (AbstractDatasetInfo): Dataset info.

        Returns:
            Dataset: Dataset instance.

        """
        if dataset_info not in self.datasets:
            self.datasets[dataset_info] = self.__create_dataset(dataset_info=dataset_info)
        return self.datasets[dataset_info]

    def write_data_in_dataset(self, dataset_info: AbstractDatasetInfo,
                                    data_dict: Dict[str, Any],
                                    data_type_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Writes data from data_dict into the dataset.

        Args:
            dataset_info (AbstractDatasetInfo): Dataset associated to namespaces.
            data_dict (Dict[str, Any]): Dict of data to be written in datasets with their values.
            data_type_dict (Dict[str, str]): Dict of data to be written in datasets with their types.

        Returns:
            Dict[str, Any]: Data dict of data names plus a DATASET_INFO field with DatasetInfo object.

        """
        self.__logger.debug(f"exporting data {data_dict.keys()} into dataset {dataset_info}")

        try:
            # Get the dataset, creates it if not exists
            dataset = self.get_dataset(dataset_info=dataset_info)

            # Write values
            dataset_values = dataset.connector.write_dataset(dataset_identifier=dataset_info,
                                                                values_to_write=data_dict,
                                                                data_types_dict=data_type_dict,
                                                                create_if_not_exists=True,
                                                                override=True)
        except DatasetGenericException as exception:
            raise DatasetGenericException(f'Error exporting dataset "{dataset_info.dataset_id}" of datasets connector "{dataset_info.connector_id}": {exception}')
        return dataset_values

    def get_path_to_dataset_data(self, dataset_info: AbstractDatasetInfo, data_name: str, data_type: str) -> str:
        """
        Gets the path/link/URI to retrieve the dataset data.

        Args:
            dataset_info (AbstractDatasetInfo): Dataset in which the data is.
            data_name (str): Data name to build the path.
            data_type (str): Type of the data in dataset.

        Returns:
            str: Path/link/URI to dataset data.

        """
        path_to_dataset_data = ""
        try:
            # Get the dataset, creates it if not exists
            dataset = self.get_dataset(dataset_info=dataset_info)

            # get path
            path_to_dataset_data = dataset.connector.build_path_to_data(dataset_identifier=dataset_info,
                                                                        data_name=data_name,
                                                                        data_type=data_type)
        except DatasetGenericException as exception:
            raise DatasetGenericException(f'Error finding path of data {data_name} into dataset "{dataset_info.dataset_id}" of datasets connector "{dataset_info.connector_id}": {exception}')
        return path_to_dataset_data

    def __create_dataset(self, dataset_info: AbstractDatasetInfo) -> Dataset:
        """
        Private method to get the connector associated to the dataset and create a Dataset object.

        Args:
            dataset_info (AbstractDatasetInfo): Dataset info.

        Returns:
            Dataset: Dataset instance.

        """
        # Gets connector
        connector = DatasetsConnectorManager.get_connector(connector_identifier=dataset_info.connector_id)

        return Dataset(dataset_info=dataset_info, connector=connector)
