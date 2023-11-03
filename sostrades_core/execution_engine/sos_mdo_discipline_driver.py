'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2023/11/03 Copyright 2023 Capgemini

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

from gemseo.core.discipline import MDODiscipline
from sostrades_core.tools.filter.filter import filter_variables_to_convert
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.sos_mdo_discipline import SoSMDODiscipline
from sostrades_core.execution_engine.data_connector.data_connector_factory import ConnectorFactory
import logging
# debug mode
from copy import deepcopy
from pandas import DataFrame
from numpy import ndarray, floating

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

class SoSMDODriverException(Exception):
    pass


class SoSMDODisciplineDriver(SoSMDODiscipline):
    def __init__(self, full_name, grammar_type, cache_type, cache_file_path, sos_wrapp, reduced_dm, disciplines, logger:logging.Logger):
        super().__init__(full_name, grammar_type, cache_type, cache_file_path, sos_wrapp, reduced_dm, logger=logger)
        self.disciplines = disciplines
