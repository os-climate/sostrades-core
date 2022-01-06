'''
Copyright 2022 Airbus SAS

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
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
"""

from os.path import basename, splitext
import yaml
from sos_trades_core.tools.yaml import Rx


class YamlModelcheckerException(Exception):
    pass


def check_yaml(yaml_schema, yaml_data, raise_on_error=False):
    """ using yaml data as object (not file or stream) to perform a check of yaml data structure
    regarding the yaml schema given as arguments

    args:
        yaml_schema (object) : yaml schema to apply
        yaml_data (object) : yaml data to check regarding yaml_schema
        raise_on_error (boolean) : if true launch an exception on error, otherwise return false

    return:
        boolean: result of the comparison
    """

    rx_factory = Rx.Factory({"register_core_types": True})

    schema = None
    try:
        schema = rx_factory.make_schema(yaml_schema)
    except Rx.SchemaError as se_ex:
        if raise_on_error:
            raise YamlModelcheckerException('{}'.format(se_ex))
        else:
            return False

    if not schema:
        return False

    try:
        schema.validate(yaml_data)
    except Rx.SchemaMismatch as sm_ex:
        if raise_on_error:
            raise YamlModelcheckerException('{}'.format(sm_ex))
        else:
            return False

    return True


def check_yaml_from_file(
        yaml_schema_file, yaml_data_file, raise_on_error=False):
    """ using yaml data as filepath to perform a check of yaml data structure
    regarding the yaml schema given as arguments

    args:
        yaml_schema_file (string) : yaml schema file to apply
        yaml_data_file (string) : yaml data file to check regarding yaml_schema
        raise_on_error (boolean) : if true launch an exception on error, otherwise return false

    return:
        boolean: result of the comparison
    """

    schema = None
    with open(yaml_schema_file) as stream:
        schema = yaml.load(stream, Loader=yaml.FullLoader)

    data = None
    with open(yaml_data_file) as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    return check_yaml(schema, data, raise_on_error)


def yaml_file_to_dictionary(yaml_files):
    """ Convert yaml file into dictionary with filename as key yaml object as value

    args:
        yaml_files (string or string array)

    return
        dictionary {filename: yaml object}
    """

    result = {}

    if not isinstance(yaml_files, list):
        yaml_files = [yaml_files]

    for yaml_file in yaml_files:

        yaml_data = open_yaml(yaml_file)
#         with open(model) as stream:
#             yaml_data = yaml.load(stream, Loader=yaml.FullLoader)

        filename_with_ext = basename(yaml_file)
        filename = splitext(filename_with_ext)[0]

        result[filename] = yaml_data

    return result


def open_yaml(yaml_file):
    """ converts yaml to dict (Simple call to yaml.load)
    """
    with open(yaml_file) as stream:
        yaml_data = yaml.load(stream, Loader=yaml.FullLoader)
        return yaml_data
