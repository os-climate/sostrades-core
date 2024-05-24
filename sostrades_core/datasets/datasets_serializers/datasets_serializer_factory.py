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
from enum import Enum

from sostrades_core.datasets.datasets_serializers.abstract_datasets_serializer import (
    AbstractDatasetsSerializer,
)
from sostrades_core.datasets.datasets_serializers.json_datasets_serializer import (
    JSONDatasetsSerializer,
)
from sostrades_core.datasets.datasets_serializers.filesystem_datasets_serializer import (
    FileSystemDatasetsSerializer,
)

from sostrades_core.tools.metaclasses.no_instance import NoInstanceMeta


class DatasetSerializerType(Enum):
    """
    Dataset serializer types enum
    """

    JSON = JSONDatasetsSerializer
    FileSystem = FileSystemDatasetsSerializer

    @classmethod
    def get_enum_value(cls, value_str):
        try:
            # Iterate through the enum members and find the one with a matching value
            return next(member for member in cls if member.name == value_str)
        except StopIteration:
            raise ValueError(f"No matching enum value found for '{value_str}'")


class DatasetsSerializerFactory(metaclass=NoInstanceMeta):
    """
    Dataset serializer factory
    """
    __logger = logging.getLogger(__name__)

    @classmethod
    def get_serializer(cls,
        serializer_type: DatasetSerializerType
    ) -> AbstractDatasetsSerializer:
        """
        Instanciate a serializer of type serializer_type with provided arguments
        Raises ValueError if type is invalid

        :param serializer_type: serializer type to instanciate
        :type serializer_type: DatasetserializerType
        """
        cls.__logger.debug(f"Instanciating serializer of type {serializer_type}")
        if not isinstance(serializer_type, DatasetSerializerType) or not issubclass(
            serializer_type.value, AbstractDatasetsSerializer
        ):
            raise ValueError(f"Unexpected serializer type {serializer_type}")
        try:
            return serializer_type.value()
        except TypeError as exc:
            raise Exception(serializer_type) from exc


