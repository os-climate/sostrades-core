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
import numpy as np
import pandas as pd

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

class Disc1(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'Disc1_nested_types',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-plane fa-fw',
        'version': '',
    }
    _maturity = 'Fake'
    # TODO: subtype descriptors !?
    DESC_IN = {

        'X_dict_df': {'type': 'dict'},
        'X_dict_dict_df': {'type': 'dict'},
        'X_dict_dict_float': {'type': 'dict'},  # NB: this is perfectly jsonifiable no need to pkl
        'X_array_string': {'type': 'array'},
        'X_array_df': {'type': 'array'},
    }

    def run(self):
        pass

