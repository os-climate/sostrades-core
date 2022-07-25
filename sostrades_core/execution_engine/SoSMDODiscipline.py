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
from scipy.sparse.lil import lil_matrix
from gemseo.utils.derivatives.derivatives_approx import DisciplineJacApprox
from gemseo.core.discipline import MDODiscipline

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
# set-up the folder where GEMSEO will look-up for new wrapps (solvers,
# grammars etc)
import os
from os.path import dirname, join

parent_dir = dirname(__file__)
GEMSEO_ADDON_DIR = "gemseo_addon"
os.environ["GEMSEO_PATH"] = join(parent_dir, GEMSEO_ADDON_DIR)

from copy import deepcopy

from pandas import DataFrame
from numpy import ndarray

from numpy import int32 as np_int32, float64 as np_float64, complex128 as np_complex128, int64 as np_int64, floating

# from gemseo.core.discipline import MDODiscipline
from gemseo.utils.compare_data_manager_tooling import dict_are_equal
from sostrades_core.api import get_sos_logger
# from gemseo.core.chain import MDOChain
from sostrades_core.execution_engine.data_connector.data_connector_factory import ConnectorFactory

from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import convert_array_into_new_type, \
    convert_new_type_into_array


class SoSMDODisciplineException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'


class SoSMDODiscipline(MDODiscipline):
    """**SoSMDODiscipline** is the class which overloads MDODiscipline
    The _run method is overloaded and new methods ( formerly from SoSDiscipline) are added

   """
    def __init__(self, full_name, grammar_type, cache_type, sos_wrapp):
        '''
        Constructor
        '''
        self.sos_wrapp = sos_wrapp
        MDODiscipline.__init__(self, name=full_name,
                               grammar_type=grammar_type,
                               cache_type=cache_type)

    def _run(self):
        self.sos_wrapp.run()
