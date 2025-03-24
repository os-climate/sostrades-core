'''
Copyright 2025 Capgemini
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
from sostrades_core.sos_wrapping.test_discs.disc_all_types import DiscAllTypes as DaT


class DiscAllTypes2(DaT):
    # ontology information
    _ontology_data = {
        'label': 'All type class with heritage',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': 'discipline used to test ontology import fix',
        'icon': 'fa-circle-user',
        'version': '',
    }
