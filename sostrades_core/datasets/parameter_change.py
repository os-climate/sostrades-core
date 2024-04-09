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
from typing import Any, Union
from dataclasses import dataclass
from datetime import datetime
from gemseo.utils.compare_data_manager_tooling import dict_are_equal


# TODO [to discuss]: move to data_manager.parameter_change (and create data_manager.data_manager) ?
@dataclass()
class ParameterChange:
    parameter_id: str
    variable_type: str  # OR type ?
    old_value: Any
    new_value: Any
    dataset_id: Union[str, None]
    connector_id: Union[str, None]
    date: datetime
