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

    VERSION_SUFFIX: str = "__@"

    @abc.abstractmethod
    def __init__(self,
                 connector_id: str,
                 mono_version_connectors_instantiation_fields: dict[str:dict[str:Any]]):
        """
        Abstract init method forcing to create a dedicated subclass for every type of multi-version connector (JSON,
        LocalFileSystem, etc.). These multi-version subclasses must overload the VERSION_TO_CLASS dict, specifying the
        mono-version connector class associated to each compatible version handled by the multi-version connector. Only
        the versions thus declared will be considered compatible with the multi-version connector. Note that, when a
        multi-version connector is registered, its associated mono-version components are not registered independently.

        Args:
            connector_id: Connector identifier for the multiversion connector
            mono_version_connectors_instantiation_fields: keyword arguments that allow to instantiate the different
                mono-version connectors.
        """
        self.connector_id = connector_id
        self.__version_connectors = {}
        for _version, _version_fields in mono_version_connectors_instantiation_fields.items():
            if _version not in self.VERSION_TO_CLASS:
                raise DatasetGenericException(f"Multi-version connector {self} does not implement version {_version}.")
            _version_class = self.VERSION_TO_CLASS[_version]
            if _version in _version_class.COMPATIBLE_DATASET_INFO_VERSION:
                # TODO: not registering the mono version connectors independently, should I?
                if self.CONNECTOR_ID in _version_fields:
                    self.__version_connectors[_version] = _version_class(**_version_fields)
                else:
                    # TODO: this custom naming for subconnectors is to review, and it is unused unless registered
                    subconnector_id = self.__get_subconnector_id(_version)
                    self.__version_connectors[_version] = _version_class(connector_id=subconnector_id,
                                                                         **_version_fields)
            else:
                raise DatasetGenericException(f"The class {_version_class.__name__} defined for version {_version} in "
                                              f"multi-version connector {self} is not compatible with {_version}.")

    def __get_subconnector_id(self, version: str):
        return self.connector_id + self.VERSION_SUFFIX + version

    @property
    def all_connectors(self):
        return self.__version_connectors.values()

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
                                                self.all_connectors))))

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
        # CLEARING

    def clear_dataset(self, dataset_id: str, version_id: str = None) -> None:
        """
        Optional utility method to remove a given dataset_id within a certain connector.

        Args:
            dataset_id (str): Identifier of the dataset to be removed
            version_id (str): Version of the dataset to be removed, if None will remove from all sub-connectors.
        """
        if version_id is None:
            for _c in self.all_connectors:
                _c.clear_dataset(dataset_id)
        elif version_id in self.VERSION_TO_CLASS:
            self.__version_connectors[version_id].clear_dataset(dataset_id)
        else:
            raise DatasetGenericException(f"Multi-version connector {self} does handle version {version_id}.")

    def clear_all_datasets(self):
        """
        Optional utility method to remove all datasets in a connector.
        """
        for _c in self.all_connectors:
            _c.clear_all_datasets()

    def clear_connector(self):
        """
        Optional utility method to completely clear a connector further than clearing all datasets, if it applies, e.g.
        by deleting the root directory of a local connector, or by deleting the database file of a json connector. It
        defaults to clear_all_datasets unless overloaded.
        """
        for _c in self.all_connectors:
            _c.clear_connector()

