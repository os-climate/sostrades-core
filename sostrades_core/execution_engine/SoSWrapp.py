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


class SoSWrappException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'


class SoSWrapp(object):
    '''**SoSWrapp** is the class from which inherits our model Wrapper
    It contains necessary information for the discipline configuration
    Its methods setup_sos_disciplines, run,... are overloaded by the discipline Wrapper

    '''
    # -- Disciplinary attributes
    DESC_IN = {}
    DESC_OUT = {}
    IO_TYPE = 'io_type'
    IO_TYPE_IN = 'in'
    IO_TYPE_OUT = 'out'
    TYPE = 'type'
    SUBTYPE = 'subtype_descriptor'
    COUPLING = 'coupling'
    VISIBILITY = 'visibility'
    LOCAL_VISIBILITY = 'Local'
    INTERNAL_VISIBILITY = 'Internal'
    SHARED_VISIBILITY = 'Shared'
    AVAILABLE_VISIBILITIES = [
        LOCAL_VISIBILITY,
        INTERNAL_VISIBILITY,
        SHARED_VISIBILITY]
    NAMESPACE = 'namespace'
    NS_REFERENCE = 'ns_reference'
    VALUE = 'value'
    DEFAULT = 'default'
    EDITABLE = 'editable'
    USER_LEVEL = 'user_level'
    STRUCTURING = 'structuring'
    POSSIBLE_VALUES = 'possible_values'
    RANGE = 'range'
    UNIT = 'unit'
    DESCRIPTION = 'description'
    NUMERICAL = 'numerical'
    META_INPUT = 'meta_input'
    OPTIONAL = 'optional'
    ORIGIN = 'model_origin'
    HEADERS = 'headers'
    COMPOSED_OF = 'composed_of'
    DISCIPLINES_DEPENDENCIES = 'disciplines_dependencies'
    VAR_NAME = 'var_name'
    VISIBLE = 'visible'
    CONNECTOR_DATA = 'connector_data'
    CACHE_TYPE = 'cache_type'
    CACHE_FILE_PATH = 'cache_file_path'

    DATA_TO_CHECK = [TYPE, UNIT, RANGE,
                     POSSIBLE_VALUES, USER_LEVEL]
    NO_UNIT_TYPES = ['bool', 'string', 'string_list']
    # Dict  ex: {'ColumnName': (column_data_type, column_data_range,
    # column_editable)}
    DATAFRAME_DESCRIPTOR = 'dataframe_descriptor'
    DATAFRAME_EDITION_LOCKED = 'dataframe_edition_locked'

    DF_EXCLUDED_COLUMNS = 'dataframe_excluded_columns'
    DEFAULT_EXCLUDED_COLUMNS = ['year', 'years']
    DISCIPLINES_FULL_PATH_LIST = 'discipline_full_path_list'

    # -- Variable types information section
    VAR_TYPE_ID = 'type'
    # complex can also be a type if we use complex step
    INT_MAP = (int, np_int32, np_int64, np_complex128)
    FLOAT_MAP = (float, np_float64, np_complex128)
    VAR_TYPE_MAP = {
        # an integer cannot be a float
        'int': INT_MAP,
        # a designed float can be integer (with python 3 no problem)
        'float': FLOAT_MAP + INT_MAP,
        'string': str,
        'string_list': list,
        'string_list_list': list,
        'float_list': list,
        'int_list': list,
        'dict_list': list,
        'array': ndarray,
        'df_dict': dict,
        'dict': dict,
        'dataframe': DataFrame,
        'bool': bool,
        'list': list
    }
    VAR_TYPE_GEMS = ['int', 'array', 'float_list', 'int_list']
    STANDARD_TYPES = [int, float, np_int32, np_int64, np_float64, bool]
    #    VAR_TYPES_SINGLE_VALUES = ['int', 'float', 'string', 'bool', 'np_int32', 'np_float64', 'np_int64']
    NEW_VAR_TYPE = ['dict', 'dataframe',
                    'string_list', 'string', 'float', 'int', 'list']

    UNSUPPORTED_GEMSEO_TYPES = []
    for type in VAR_TYPE_MAP.keys():
        if type not in VAR_TYPE_GEMS and type not in NEW_VAR_TYPE:
            UNSUPPORTED_GEMSEO_TYPES.append(type)

    # Warning : We cannot put string_list into dict, all other types inside a dict are possiblr with the type dict
    # df_dict = dict , string_dict = dict, list_dict = dict
    TYPE_METADATA = "type_metadata"
    DEFAULT = 'default'
    POS_IN_MODE = ['value', 'list', 'dict']

    AVAILABLE_DEBUG_MODE = ["", "nan", "input_change",
                            "linearize_data_change", "min_max_grad", "min_max_couplings", "all"]

    # -- status section

    # -- Maturity section
    possible_maturities = [
        'Fake',
        'Research',
        'Official',
        'Official Validated']
    dict_maturity_ref = dict(zip(possible_maturities,
                                 [0] * len(possible_maturities)))

    NUM_DESC_IN = {
        'linearization_mode': {TYPE: 'string', DEFAULT: 'auto',  # POSSIBLE_VALUES: list(MDODiscipline.AVAILABLE_MODES),
                               NUMERICAL: True},
        CACHE_TYPE: {TYPE: 'string', DEFAULT: 'None',
                     # POSSIBLE_VALUES: ['None', MDODiscipline.SIMPLE_CACHE],
                     # ['None', MDODiscipline.SIMPLE_CACHE, MDODiscipline.HDF5_CACHE, MDODiscipline.MEMORY_FULL_CACHE]
                     NUMERICAL: True,
                     STRUCTURING: True},
        CACHE_FILE_PATH: {TYPE: 'string', DEFAULT: '', NUMERICAL: True, OPTIONAL: True, STRUCTURING: True},
        'debug_mode': {TYPE: 'string', DEFAULT: '', POSSIBLE_VALUES: list(AVAILABLE_DEBUG_MODE),
                       NUMERICAL: True, 'structuring': True}
    }

    # -- grammars
    SOS_GRAMMAR_TYPE = "SoSSimpleGrammar"

    # -- status
    STATUS_VIRTUAL = "VIRTUAL"
    STATUS_PENDING = "PENDING"
    STATUS_DONE = "DONE"
    STATUS_RUNNING = "RUNNING"
    STATUS_FAILED = "FAILED"
    STATUS_CONFIGURE = 'CONFIGURE'
    STATUS_LINEARIZE = 'LINEARIZE'

    def __init__(self, sos_name):
        '''
        Constructor
        '''
        self.sos_name = sos_name

    def set_up_sos_discipline(self,proxy):  # type: (...) -> None
        """Define the set_up_sos_discipline of its proxy

        To be overloaded by subclasses.
        """
        raise NotImplementedError()

    def run(self,proxy):  # type: (...) -> None
        """Define the run of the discipline

        To be overloaded by subclasses.
        """
        raise NotImplementedError()

