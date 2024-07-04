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
from typing import Any

import numpy as np
import pandas as pd

from sostrades_core.datasets.datasets_serializers.json_datasets_serializer import (
    JSONDatasetsSerializer,
)


class BigQueryDatasetsSerializer(JSONDatasetsSerializer):
    """
    Specific dataset serializer for dataset in json format
    """
    def __init__(self):
        super().__init__()
        self.__logger = logging.getLogger(__name__)

    def _serialize_dataframe(self, data_value, data_name):
        return data_value