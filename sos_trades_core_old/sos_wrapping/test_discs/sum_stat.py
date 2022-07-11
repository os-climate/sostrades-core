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
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
import itertools
import numpy as np
import pandas as pd


class Sumstat(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'Sumstat discipline',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }

    _maturity = 'Fake'

    DESC_IN = {
        'stat_A': {'type': 'float', 'default': 1.3, 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_sum_stat'},
        'stat_B': {'type': 'float', 'default': 1.3, 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_sum_stat'},
        'stat_C': {'type': 'float', 'default': 1.3, 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_sum_stat'}
    }
    DESC_OUT = {
        'sum_stat': {'type': 'float', 'unit': '-', 'visibility': 'Shared',
                     'namespace': 'ns_sum_stat'}
    }

    def run(self):

        input_dict = self.get_sosdisc_inputs()
        sum_stat = input_dict['stat_A'] + \
            input_dict['stat_B'] + input_dict['stat_C']
        dict_values = {'sum_stat': sum_stat}
        self.store_sos_outputs_values(dict_values)
