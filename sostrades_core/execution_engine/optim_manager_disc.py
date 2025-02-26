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
import warnings

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

warnings.simplefilter(action='ignore', category=FutureWarning)


class OptimManagerDisc(SoSWrapp):
    """Constraints aggregation discipline"""

    # ontology information
    _ontology_data = {
        'label': 'Optim Function Manager',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-chart-line fa-fw',
        'version': '',
    }

    def __init__(self, sos_name, logger: logging.Logger):
        '''Constructor'''
        super().__init__(sos_name=sos_name, logger=logger)
