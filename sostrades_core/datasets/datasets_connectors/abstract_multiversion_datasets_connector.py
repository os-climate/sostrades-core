'''
Copyright 2024 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain aD copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Type
from itertools import chain
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (AbstractDatasetsConnector,
                                                                                     DatasetGenericException)

if TYPE_CHECKING:
    from sostrades_core.datasets.dataset_info.abstract_dataset_info import AbstractDatasetInfo


class AbstractMultiVersionDatasetsConnector(AbstractDatasetsConnector, abc.ABC):
    """
    Abstract class to inherit in order to build specific datasets connector
    """
    __logger = logging.getLogger(__name__)

    COMPATIBLE_DATASET_INFO_VERSION = None  # sanity to assure variable is not used in multi-version classes
    # version to connector class mapping, overloaded in subclasses
    VERSION_TO_CLASS: dict[str:Type[AbstractDatasetsConnector]] = {}

    @abc.abstractmethod
    def __init__(self, connector_id: str, **connector_instantiation_fields):
        """
        Abstract init method forcing to create a dedicated subclass for every type of multi-version connector, declaring
        in them the VERSION_TO_CLASS with the mono-version connector classes associated to each compatible version
        handled by the multi-version connector.
            connector_id (str): Connector identifier for the multiversion connector
            **connector_instantiation_fields: keyword arguments that allow to instantiate the different mono-version
                connectors. They are assumed to be the same for all mono-version classes of a same type of connector.
        """
        self.__version_connectors = {}
        for _version, _version_class in self.VERSION_TO_CLASS.items():
            if _version in _version_class.__version_connectors:
                self.__version_connectors[_version] = _version_class(connector_id=connector_id,
                                                                                **connector_instantiation_fields)
            else:
                raise DatasetGenericException(f"The class defined for version {_version} is not compatible.")

    @property
    def compatible_dataset_info_version(self):
        return set(self.__version_connectors.keys())

    def version_connector(self, dataset_identifier: AbstractDatasetInfo) -> AbstractDatasetsConnector:
        """
        Method that gets the appropriate version connector from a datasetinfo.
        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
        Returns:
            version_connector (AbstractDatasetsConnector): connector of the appropriate version
        """
        return self.__version_connectors[dataset_identifier.version_id]

    def _get_values(self, dataset_identifier: AbstractDatasetInfo, data_to_get: Dict[str, str]) -> Dict[str, Any]:
        """
        Get a list of data from a specific API. Delegates to public method of corresponding version connector.

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            data_to_get (Dict[str, str]): dict of data name and type of data to get {name: type}

        Returns:
            Dict[str, Any]: Retrieved data
        """

        return self.version_connector(dataset_identifier).get_values(dataset_identifier=dataset_identifier,
                                                                     data_to_get=data_to_get)

    def _write_values(self, dataset_identifier: AbstractDatasetInfo, values_to_write: Dict[str, Any],
                      data_types_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Protected method to write data to a specific API. Delegates to public method of corresponding version connector.

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            values_to_write (Dict[str, Any]): dict of data to write {name: value}
            data_types_dict (Dict[str, str]): dict of data type {name: type}

        Returns:
            Dict[str, Any]: Written data
        """
        return self.version_connector(dataset_identifier).write_values(dataset_identifier=dataset_identifier,
                                                                       values_to_write=values_to_write,
                                                                       data_types_dict=data_types_dict)

    def _get_values_all(self, dataset_identifier: AbstractDatasetInfo, data_types_dict: Dict[str, str]
                        ) -> Dict[str, Any]:
        """
        Protected Abstract method to get all values from a dataset for a specific API. Delegates to public method of
        corresponding version connector.

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            data_types_dict (Dict[str, str]): dict of data types {name: type}

        Returns:
            Dict[str, Any]: All values from the dataset
        """
        return self.version_connector(dataset_identifier).get_values_all(dataset_identifier=dataset_identifier,
                                                                         data_types_dict=data_types_dict)

    def get_datasets_available(self) -> List[AbstractDatasetInfo]:
        """
        Abstract method to get all available datasets for a specific API. Maps the call to all instantiated compatible
        version connectors and chains the output in a single list, removing duplicates.

        Returns:
            List[AbstractDatasetInfo]: List of available datasets
        """
        return list(set(chain.from_iterable(map(lambda _c: _c.get_datasets_available(),
                                                self.__version_connectors.values()))))

    def _write_dataset(self, dataset_identifier: AbstractDatasetInfo, values_to_write: Dict[str, Any],
                       data_types_dict: Dict[str, str], create_if_not_exists: bool = True, override: bool = False
                       ) -> Dict[str, Any]:
        """
        Protected method to overload in order to write a dataset to a specific API. Delegates to public method of
        corresponding version connector.

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            values_to_write (Dict[str, Any]): dict of data to write {name: value}
            data_types_dict (Dict[str, str]): dict of data types {name: type}
            create_if_not_exists (bool, optional): Create the dataset if it does not exist. Defaults to True.
            override (bool, optional): Override dataset if it exists. Defaults to False.

        Returns:
            Dict[str, Any]: Written data
        """
        return self.version_connector(dataset_identifier).write_dataset(dataset_identifier=dataset_identifier,
                                                                        values_to_write=values_to_write,
                                                                        data_types_dict=data_types_dict,
                                                                        create_if_not_exists=create_if_not_exists,
                                                                        override=override)

    def _build_path_to_data(self, dataset_identifier: AbstractDatasetInfo, data_name: str, data_type: str) -> str:
        """
        Method that can be overloaded in order to build the path to a dataset data for a specific API. Delegates to
        public method of corresponding version connector.

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier into connector
            data_name (str): data in dataset
            data_type (str): type of the data in dataset

        Returns:
            str: Path/URL/URI to find the dataset data
        """
        return self.version_connector(dataset_identifier).build_path_to_data(dataset_identifier=dataset_identifier,
                                                                             data_name=data_name,
                                                                             data_type=data_type)
