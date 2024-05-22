'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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

import inspect

from sostrades_core.execution_engine.archi_builder import ArchiBuilder
from sostrades_core.execution_engine.proxy_discipline_builder import (
    ProxyDisciplineBuilder,
)
from sostrades_core.execution_engine.sos_builder import SoSBuilder


def get_ns_list_in_builder_list(builder_list):
    ns_list = []
    if not isinstance(builder_list, list):
        raise Exception(f'The argument of get_ns_list_in_builder_list is not a list : {builder_list}')

    for builder in builder_list:
        ns_sublist = get_ns_list_in_sub_builder(builder)
        ns_list.extend(ns_sublist)
    return list(set(ns_list))

def get_ns_list_in_sub_builder(builder):

    if not isinstance(builder, SoSBuilder):
        raise Exception(f'Need a builder to extract namespaces not :{type(builder)}')
    ns_list = []
    if builder.cls.DESC_IN is not None:
        ns_list_in = [value_dict['namespace'] for value_dict in builder.cls.DESC_IN.values() if
                      'namespace' in value_dict]
        ns_list.extend(ns_list_in)
    if builder.cls.DESC_OUT is not None:
        ns_list_out = [value_dict['namespace'] for value_dict in builder.cls.DESC_OUT.values() if
                       'namespace' in value_dict]
        ns_list.extend(ns_list_out)
    if hasattr(builder.cls, 'DYNAMIC_VAR_NAMESPACE_LIST'):
        ns_list.extend(builder.cls.DYNAMIC_VAR_NAMESPACE_LIST)

    if ProxyDisciplineBuilder in inspect.getmro(builder.cls):
        if builder.cls == ArchiBuilder:
            print(
                'Namespaces in architecture df of an archibuilder are not found to be updated : Use a scatter_map with ns_to_update if needed')
        else:
            builder_list_to_build = builder.args['cls_builder']
            for sub_builder in builder_list_to_build:
                ns_sub_list = get_ns_list_in_sub_builder(sub_builder)
                ns_list.extend(ns_sub_list)

    return ns_list
