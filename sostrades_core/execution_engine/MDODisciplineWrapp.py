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
from sostrades_core.execution_engine.SoSMDODiscipline import SoSMDODiscipline

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


class SoSWrappException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'


class MDODisciplineWrapp(object):
    '''**MDODisciplineWrapp** is the interface to create MDODiscipline from sostrades or gemseo objects


    '''

    def __init__(self, name, wrapper, wrapping_mode='SoSTrades'):
        '''
        Constructor
        '''
        self.name = name
        self.wrapping_mode = wrapping_mode
        self.wrapper = wrapper(name)

    def get_input_data_names(self, filtered_inputs=False):  # type: (...) -> List[str]
        """Return the names of the input variables.

        Returns:
            The names of the input variables.
        """
        return self.mdo_discipline.get_input_data_names(filtered_inputs)

    def get_output_data_names(self, filtered_outputs=False):  # type: (...) -> List[str]
        """Return the names of the output variables.

        Returns:
            The names of the input variables.
        """
        return self.mdo_discipline.get_output_data_names(filtered_outputs)

    def set_up_sos_discipline(self, proxy):  # type: (...) -> None
        """Define setup

        """
        if self.wrapper is not None:
            self.wrapper.set_up_sos_discipline(proxy)

    def create_gemseo_discipline(self):  # type: (...) -> None
        """ MDODiscipline instanciation

        """
        if self.wrapping_mode == 'SoSTrades':
            self.mdo_discipline = SoSMDODiscipline(self.sos_name, self.wrapper)
        else:
            # self.mdo_discipline = create_discipline(self.sos_name)
            pass

    def create_wrapp(self):  # type: (...) -> None
        """ SoSWrapp instanciation

        """
        if self.wrapping_mode == 'SoSTrades':
            # self.wrapper = SoSMDODiscipline(self.sos_name,self.wrapper)
            pass
        else:
            # self.mdo_discipline = create_discipline(self.sos_name)
            pass

    def execute(self):
        """ Discipline Execution
	    """

        return self.mdo_discipline.execute()
