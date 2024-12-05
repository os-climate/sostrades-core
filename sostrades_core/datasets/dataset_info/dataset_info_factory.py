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

import logging
import re
from enum import Enum
from warnings import warn

from sostrades_core.datasets.dataset_info.dataset_info_v0 import DatasetInfoV0
from sostrades_core.datasets.dataset_info.dataset_info_v1 import DatasetInfoV1
from sostrades_core.tools.metaclasses.no_instance import NoInstanceMeta


class DatasetInfoSerializerVersion(Enum):
    """
    DatasetInfo version enum
    """

    V0 = DatasetInfoV0
    V1 = DatasetInfoV1

    @classmethod
    def get_enum_value(cls, value_str: str) -> DatasetInfoSerializerVersion:
        """
        Get the enum value corresponding to the given string.

        Args:
            value_str (str): The string representation of the enum value.

        Returns:
            DatasetInfoSerializerVersion: The corresponding enum value.

        Raises:
            ValueError: If no matching enum value is found.
        """
        try:
            # Iterate through the enum members and find the one with a matching value
            return next(member for member in cls if member.name == value_str.upper())
        except StopIteration:
            raise ValueError(f"No matching enum value found for '{value_str}'")


class DatasetInfoFactory(metaclass=NoInstanceMeta):
    """
    Dataset info factory
    """
    __logger = logging.getLogger(__name__)

    @classmethod
    def get_dataset_info_version(cls, dataset_mapping_key: str) -> DatasetInfoSerializerVersion:
        """
        Instantiate a DatasetInfo from the version of dataset_mapping_key.
        Raises VersionNotKnownException if type is invalid.

        Args:
            dataset_mapping_key (str): Key in datasetMapping: version|connector_id|dataset_id...

        Returns:
            DatasetInfoSerializerVersion: The corresponding DatasetInfoSerializerVersion enum value.
        """
        # check if the key starts with V0 or V1 (or v0 or v1)
        version_pattern = r"^([Vv][\d])\|"
        match = re.match(version_pattern, dataset_mapping_key)
        version = DatasetInfoSerializerVersion.V0
        if match:
            version = DatasetInfoSerializerVersion.get_enum_value(match.group(1))
        else:
            warn("No version in dataset info is tolerated for now but will be deprecated in future versions", UserWarning)  # noqa: B028

        return version
