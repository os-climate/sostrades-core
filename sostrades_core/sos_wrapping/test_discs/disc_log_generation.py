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
import time
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline


class DiscLogGeneration(SoSWrapp):
    """
    Discipline to generate logs
    """
    # ontology information
    _ontology_data = {
        'label': 'DiscLogGeneration',
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
        'log_lines': {'type': 'int', 'default': 1},
        'wait_time_s': {'type': 'int', 'default': 1}
    }
    DESC_OUT = {
        'out': {'type': 'int'},
    }

    def run(self):
        start_ts = time.time()
        log_lines = self.get_sosdisc_inputs('log_lines')
        wait_time_s = self.get_sosdisc_inputs('wait_time_s')
        for i in range(log_lines):
            self.logger.warning("Sample_log | " * 100)
        time.sleep(wait_time_s)

        spent = time.time() - start_ts
        self.logger.info(f"Spent {spent}s")

        self.store_sos_outputs_values({'out': 1})