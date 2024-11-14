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


def create_data_key(disc_model_name: str, io_type: str, variable_name: str) -> str:
    '''
    Create ontology key to identify a variable
    :param disc_name: discipline model name full path in witch is the variable
    :param io_type: type in or out of the variable
    :param variable_name: name of the variable

    :return: the ontology key: {disc_model_name}_{io_type}put_{variable_name}
    '''
    io_type = io_type.lower()
    return f'{disc_model_name}_{io_type}put_{variable_name}'
