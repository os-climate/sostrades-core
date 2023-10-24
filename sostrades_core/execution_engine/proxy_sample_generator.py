'''
Copyright 2023 Capgemini

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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from copy import copy
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.ns_manager import NS_SEP


class ProxySampleGeneratorException(Exception):
    pass


class ProxySampleGenerator(ProxyDiscipline):
    '''
    Class that gather output data from a scatter discipline
    '''

    # ontology information
    _ontology_data = {
        'label': 'Sample Generator',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-outdent fa-fw',
        'version': '',
    }
