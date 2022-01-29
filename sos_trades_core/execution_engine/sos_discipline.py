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

from six import string_types
from functools import reduce
from copy import deepcopy
from pandas import DataFrame
from numpy import ndarray, append, arange, delete, array
from numpy import int32 as np_int32, float64 as np_float64, complex128 as np_complex128, int64 as np_int64, floating
from numpy import min as np_min, max as np_max

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.derivatives.derivatives_approx import DisciplineJacApprox
from sos_trades_core.sos_processes.compare_data_manager_tooling import compare_dict
from sos_trades_core.api import get_sos_logger
from gemseo.core.chain import MDOChain
from sos_trades_core.execution_engine.data_connector.data_connector_factory import ConnectorFactory


class SemMock():
    """ Mock class to override semaphore behaviour into MDODiscipline class"""

    def __init__(self, initial_value):
        """ Default contructor

            :params: initial semaphore mock value
            :type: int

        """
        self.__value = initial_value

    def __enter__(self):
        """ Context manager entry point methods 

            It makes the class works using 'with' statments
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Context manager exit point methods 

            call when exited 'with' statment
        """
        pass

    def get_lock(self):
        """ Semaphore main behaviour methods
        """
        return self

    @property
    def value(self):
        """ Semaphore value accessor """
        return self.__value

    @value.setter
    def value(self, value):
        """ Semaphore value accessor """
        self.__value = value


class SoSDisciplineException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'


class SoSDiscipline(MDODiscipline):
    '''**SoSDiscipline** is the :class:`~gemseo.core.discipline.MDODiscipline`
    interfacing Model disciplines and Gemseo generic discipline

    Use the following MDODiscipline methods:
        get_data_list_from_dict: to get input or output values
    '''
    # -- Disciplinary attributes
    DESC_IN = None
    DESC_OUT = None
    IO_TYPE = 'io_type'
    IO_TYPE_IN = 'in'
    IO_TYPE_OUT = 'out'
    TYPE = 'type'
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
    }
    VAR_TYPE_GEMS = ['int', 'array', 'float_list', 'int_list']
    STANDARD_TYPES = [int, float, np_int32, np_int64, np_float64, bool]
    #    VAR_TYPES_SINGLE_VALUES = ['int', 'float', 'string', 'bool', 'np_int32', 'np_float64', 'np_int64']
    NEW_VAR_TYPE = ['dict', 'dataframe',
                    'string_list', 'string', 'float', 'int']
    # Warning : We cannot put string_list into dict, all other types inside a dict are possiblr with the type dict
    # df_dict = dict , string_dict = dict, list_dict = dict
    TYPE_METADATA = "type_metadata"
    DEFAULT = 'default'
    POS_IN_MODE = ['value', 'list', 'dict']

    # -- status section
    STATUS_CONFIGURE = 'CONFIGURE'

    # -- Maturity section
    possible_maturities = [
        'Fake',
        'Research',
        'Official',
        'Official Validated']
    dict_maturity_ref = dict(zip(possible_maturities,
                                 [0] * len(possible_maturities)))

    NUM_DESC_IN = {
        'linearization_mode': {TYPE: 'string', DEFAULT: 'auto', POSSIBLE_VALUES: list(MDODiscipline.AVAILABLE_MODES),
                               NUMERICAL: True},
        'cache_type': {TYPE: 'string', DEFAULT: MDODiscipline.SIMPLE_CACHE,
                       POSSIBLE_VALUES: [MDODiscipline.SIMPLE_CACHE, MDODiscipline.HDF5_CACHE,
                                         MDODiscipline.MEMORY_FULL_CACHE],
                       NUMERICAL: True},
        'cache_file_path': {TYPE: 'string', NUMERICAL: True, OPTIONAL: True},
    }

    # -- grammars
    SOS_GRAMMAR_TYPE = "SoSSimpleGrammar"

    def __init__(self, sos_name, ee):
        '''
        Constructor
        '''
        # Enable not a number check in execution result and jacobian result
        # Be carreful that impact greatly calculation performances
        self._reload(sos_name, ee)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.Discipline')
        self.model = None
        self.father_builder = None
        self.father_executor = None

    def set_father_executor(self, father_executor):
        self.father_executor = father_executor

    def _reload(self, sos_name, ee):
        ''' reload object, eventually with coupling_namespace
        '''

        # ------------DEBUG VARIABLES----------------------------------------
        self.nan_check = False
        self.check_if_input_change_after_run = False
        self.check_linearize_data_changes = True
        self.check_min_max_gradients = False
        # ----------------------------------------------------

        # -- Base disciplinary attributes
        self.jac_boundaries = {}
        self.disc_id = None
        self.sos_name = sos_name
        self.ee = ee
        self.dm = self.ee.dm

        self.ee.ns_manager.create_disc_ns_info(self)

        if not hasattr(self, 'is_sos_coupling'):
            self.is_sos_coupling = False
        self.is_optim_scenario = False
        self.is_parallel = False
        self.is_specific_driver = False

        # -- Sub-disciplines attributes
        self.built_sos_disciplines = []
        self.sos_disciplines = None
        self.in_checkjac = False
        self.reset_sos_disciplines()
        # -- Maturity attribute
        self._maturity = self.get_maturity()
        self._is_configured = False
        MDODiscipline.__init__(
            self, sos_name, grammar_type=self.SOS_GRAMMAR_TYPE)
        # Update status attribute and data manager

        # -- disciplinary data attributes
        self.inst_desc_in = None  # desc_in of instance used to add dynamic inputs
        self.inst_desc_out = None  # desc_out of instance used to add dynamic outputs
        self._data_in = None
        self._data_out = None
        self._structuring_variables = None
        self.reset_data()

        # Add the discipline in the dm and get its unique disc_id (was in the
        # configure)
        self._set_dm_disc_info()

        # update discipline status to CONFIGURE
        self._update_status_dm(self.STATUS_CONFIGURE)

    def init_execution(self):
        """
        To be used to store additional attributes for wrapping
        """
        pass

    def get_shared_namespace_list(self, data_dict):
        '''
        Get the list of namespaces defined in the data_in or data_out when the visibility of the variable is shared
        '''
        shared_namespace_list = []

        for item in data_dict.values():
            self.__append_item_namespace(item, shared_namespace_list)

        return list(set(shared_namespace_list))

    def __append_item_namespace(self, item, ns_list):
        '''
        Append the namespace if the visibility is shared
        '''
        if self.VISIBILITY in item and item[self.VISIBILITY] == self.SHARED_VISIBILITY:
            ns_list.append(item[self.NAMESPACE])

    def get_data_io_dict(self, io_type):
        '''
        Get the data_in or the data_out depending on the io_type
        '''
        if io_type == self.IO_TYPE_IN:
            return self.get_data_in()
        elif io_type == self.IO_TYPE_OUT:
            return self.get_data_out()
        else:
            raise Exception(
                f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]')

    def get_data_io_dict_keys(self, io_type):
        '''
        Get the data_in or the data_out  keys depending on the io_type
        '''
        if io_type == self.IO_TYPE_IN:
            return self.get_data_in().keys()
        elif io_type == self.IO_TYPE_OUT:
            return self.get_data_out().keys()
        else:
            raise Exception(
                f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]')

    def get_data_io_from_key(self, io_type, var_name):
        '''
        Return the namespace and the data_in/data_out of the variable name
        '''
        data_dict_list = [d for k, d in self.get_data_io_dict(
            io_type).items() if k == var_name]
        if len(data_dict_list) != 1:
            raise Exception(
                f'No key matching with variable name {var_name} in the data_{io_type}')

        return data_dict_list[0]

    def get_variable_name_from_ns_key(self, io_type, ns_key):
        return self.get_data_io_dict(io_type)[ns_key][self.VAR_NAME]

    def reload_io(self):
        '''
        Create the data_in and data_out of the discipline with the DESC_IN/DESC_OUT, inst_desc_in/inst_desc_out
        and initialize GEMS grammar with it (with a filter for specific variable types)
        '''
        # set input/output data descriptions if data_in and data_out are empty
        self.create_data_io_from_desc_io()

        # update data_in/data_out with inst_desc_in/inst_desc_out
        self.update_data_io_with_inst_desc_io()

    def update_dm_with_data_dict(self, data_dict):
        self.dm.update_with_discipline_dict(
            self.disc_id, data_dict)

    def update_gems_grammar_with_data_io(self):
        # Remove unavailable GEMS type variables before initialize
        # input_grammar
        if not self.is_sos_coupling:
            filtered_data_in = self.__filter_couplings_for_gems(
                self.IO_TYPE_IN)
            filtered_data_out = self.__filter_couplings_for_gems(
                self.IO_TYPE_OUT)
            self.init_gems_grammar(filtered_data_in, self.IO_TYPE_IN)
            self.init_gems_grammar(filtered_data_out, self.IO_TYPE_OUT)

    def create_data_io_from_desc_io(self):
        '''
        Create data_in and data_out from DESC_IN and DESC_OUT if empty
        '''
        if self._data_in == {}:
            self._data_in = deepcopy(self.DESC_IN) or {}
            self.set_shared_namespaces_dependencies(self._data_in)
            self._data_in = self._prepare_data_dict(self.IO_TYPE_IN)
            self.update_dm_with_data_dict(self._data_in)

        # Deal with numerical parameters inside the sosdiscipline
        self.add_numerical_param_to_data_in()

        if self._data_out == {}:
            self._data_out = deepcopy(self.DESC_OUT) or {}
            self.set_shared_namespaces_dependencies(self._data_out)
            self._data_out = self._prepare_data_dict(self.IO_TYPE_OUT)
            self.update_dm_with_data_dict(self._data_out)

    def add_numerical_param_to_data_in(self):
        '''
        Add numerical parameters to the data_in
        '''
        num_data_in = deepcopy(self.NUM_DESC_IN)
        num_data_in = self._prepare_data_dict(
            self.IO_TYPE_IN, data_dict=num_data_in)
        self._data_in.update(num_data_in)
        self.update_dm_with_data_dict(num_data_in)

    def update_data_io_with_inst_desc_io(self):
        '''
        Update data_in and data_out with inst_desc_in and inst_desc_out
        '''
        new_inputs = {}
        new_outputs = {}
        #         modified_inputs = {}
        #         modified_outputs = {}

        for key, value in self.inst_desc_in.items():
            if not key in self._data_in.keys():
                new_inputs[key] = value
        #             else:
        #                 if self._data_in[key][self.NAMESPACE] != value[self.NAMESPACE] and hasattr(self, 'instance_list'):
        #                     modified_inputs[key] = value

        for key, value in self.inst_desc_out.items():
            if not key in self._data_out.keys():
                new_outputs[key] = value
        #             else:
        #                 if self._data_out[key][self.NAMESPACE] != value[self.NAMESPACE] and hasattr(self, 'instance_list'):
        #                     modified_outputs[key] = value
        # add new inputs from inst_desc_in to data_in
        if len(new_inputs) > 0:
            self.set_shared_namespaces_dependencies(new_inputs)
            completed_new_inputs = self._prepare_data_dict(
                self.IO_TYPE_IN, new_inputs)
            self.update_dm_with_data_dict(
                completed_new_inputs)
            self._data_in.update(completed_new_inputs)

        # add new outputs from inst_desc_out to data_out
        if len(new_outputs) > 0:
            self.set_shared_namespaces_dependencies(new_outputs)
            completed_new_outputs = self._prepare_data_dict(
                self.IO_TYPE_OUT, new_outputs)
            self.update_dm_with_data_dict(
                completed_new_outputs)
            self._data_out.update(completed_new_outputs)

    #         if len(modified_inputs) > 0:
    #             completed_modified_inputs = self._prepare_data_dict(
    #                 self.IO_TYPE_IN, modified_inputs)
    #             self._data_in.update(completed_modified_inputs)

    def init_gems_grammar(self, data_keys, io_type):
        '''
        Init Gems grammar with keys from a data_in/out dict
        io_type specifies 'IN' or 'OUT'
        '''
        self._init_grammar_with_keys(data_keys, io_type)

    def reset_sos_disciplines(self):
        ''' Empty sos_disciplines list
        '''
        self.sos_disciplines = []

    def get_sos_diciplines_ids(self):
        return [disc.name for disc in self.sos_disciplines]

    def get_sos_disciplines(self):

        return self.sos_disciplines

    def get_sub_sos_disciplines(self, disc_list=None):
        ''' recursively returns all subdisciplines
        '''
        if disc_list is None:
            disc_list = []
        for disc in self.sos_disciplines:
            disc_list.append(disc)
            disc.get_sub_sos_disciplines(disc_list)
        return disc_list

    @property
    def ordered_disc_list(self):
        '''
         Property to obtain the ordered list of disciplines by default for
         a sos_discipline it is the order of sos_disciplines
        '''
        return self.sos_disciplines

    def add_discipline(self, disc):
        ''' add a discipline
        '''
        self.sos_disciplines.append(disc)
        disc.set_father_executor(self)
        # self._check_if_duplicated_disc_names()

    def add_discipline_list(self, disc_list):
        ''' add a list of disciplines
        '''
        for disc in disc_list:
            self.add_discipline(disc)

    def set_shared_namespaces_dependencies(self, data_dict):
        '''
        Set dependencies of shared inputs and outputs in ns_manager
        '''
        shared_namespace_list = self.get_shared_namespace_list(
            data_dict)
        self.ee.ns_manager.add_dependencies_to_shared_namespace(
            self, shared_namespace_list)

    def add_variables(self, data_dict, io_type, clean_variables=True):
        '''
        Add dynamic inputs/outputs in ins_desc_in/ints_desc_out and remove old dynamic inputs/outputs
        '''
        if io_type == self.IO_TYPE_IN:
            variables_to_remove = [
                key for key in self.inst_desc_in if key not in data_dict]
            self.inst_desc_in.update(data_dict)
        elif io_type == self.IO_TYPE_OUT:
            variables_to_remove = [
                key for key in self.inst_desc_out if key not in data_dict]
            self.inst_desc_out.update(data_dict)

        if clean_variables:
            self.clean_variables(variables_to_remove, io_type)

    def add_inputs(self, data_dict, clean_inputs=True):
        '''
        Add dynamic inputs
        '''
        self.add_variables(data_dict, self.IO_TYPE_IN,
                           clean_variables=clean_inputs)

    def add_outputs(self, data_dict, clean_outputs=True):
        '''
        Add dynamic outputs
        '''
        self.add_variables(data_dict, self.IO_TYPE_OUT,
                           clean_variables=clean_outputs)

    def clean_variables(self, var_name_list, io_type):
        '''
        Remove variables from data_in/data_out, inst_desc_in/inst_desc_out and datamanger
        '''
        for var_name in var_name_list:
            if io_type == self.IO_TYPE_IN:
                self.ee.dm.remove_keys(
                    self.disc_id, self.get_var_full_name(var_name, self._data_in))
                del self._data_in[var_name]
                del self.inst_desc_in[var_name]
            elif io_type == self.IO_TYPE_OUT:
                self.ee.dm.remove_keys(
                    self.disc_id, self.get_var_full_name(var_name, self._data_out))
                del self._data_out[var_name]
                del self.inst_desc_out[var_name]
            if var_name in self._structuring_variables:
                del self._structuring_variables[var_name]

    def update_default_value(self, var_name, io_type, new_default_value):
        '''
        Update DEFAULT and VALUE of var_name in data_io
        '''
        if var_name in self.get_data_io_dict(io_type):
            self.get_data_io_dict(
                io_type)[var_name][self.DEFAULT] = new_default_value
            self.get_data_io_dict(
                io_type)[var_name][self.VALUE] = new_default_value

    # -- Configure handling
    def configure(self):
        '''
        Configure the SoSDiscipline
        '''
        if self.check_structuring_variables_changes():
            self.set_structuring_variables_values()

        self.setup_sos_disciplines()

        self.reload_io()

        self.set_numerical_parameters()
        # update discipline status to CONFIGURE
        self._update_status_dm(self.STATUS_CONFIGURE)

        self.set_configure_status(True)

    def set_numerical_parameters(self):
        '''
        Set numerical parameters of the sos_discipline defined in the NUM_DESC_IN 
        '''
        self.linearization_mode = self.get_sosdisc_inputs('linearization_mode')
        cache_type = self.get_sosdisc_inputs('cache_type')
        cache_file_path = self.get_sosdisc_inputs('cache_file_path')

        if cache_type == MDOChain.HDF5_CACHE and cache_file_path is None:
            raise Exception(
                'if the cache type is set to HDF5Cache the cache_file path must be set')
        elif cache_type != self._cache_type or cache_file_path != self._cache_file_path:
            self.set_cache_policy(cache_type=cache_type,
                                  cache_hdf_file=cache_file_path)

    def setup_sos_disciplines(self):
        '''
        Method to be overloaded to add dynamic inputs/outputs using add_inputs/add_outputs methods.
        If the value of an input X determines dynamic inputs/outputs generation, then the input X is structuring and the item 'structuring':True is needed in the DESC_IN
        DESC_IN = {'X': {'structuring':True}}
        '''
        pass

    # -- cache handling
    def clear_cache(self):
        # -- Need to clear cache for gradients analysis
        self.cache.clear()
        for discipline in self.sos_disciplines:
            discipline.clear_cache()

    # -- data handling section
    def reset_data(self):
        self.inst_desc_in = {}
        self.inst_desc_out = {}
        self._data_in = {}
        self._data_out = {}
        self._structuring_variables = {}

    def get_data_in(self):
        return self._data_in

    def get_data_out(self):
        return self._data_out

    def _update_with_values(self, to_update, update_with):
        ''' update <to_update> 'value' field with <update_with>
        '''
        to_update_dm = {}
        ns_update_with = {}
        for k, v in update_with.items():
            ns_update_with[k] = v
            # -- Crash if key does not exist in to_update
            for key in ns_update_with.keys():
                if to_update[key][self.VISIBILITY] == self.INTERNAL_VISIBILITY:
                    raise Exception(
                        f'It is not possible to update the variable {key} which has a visibility Internal')
                else:
                    to_update_dm[self.get_var_full_name(
                        key, to_update)] = ns_update_with[key]

        self.dm.set_values_from_dict(to_update_dm)

    def get_ns_reference(self, visibility, namespace=None):
        '''Get namespace reference by consulting the namespace_manager 
        '''
        ns_manager = self.ee.ns_manager

        if visibility == self.LOCAL_VISIBILITY or visibility == self.INTERNAL_VISIBILITY:
            return ns_manager.get_local_namespace(self)

        elif visibility == self.SHARED_VISIBILITY:
            return ns_manager.get_shared_namespace(self, namespace)

    def apply_visibility_ns(self, io_type):
        '''
        Consult the namespace_manager to apply the namespace
        depending on the variable visibility
        '''
        dict_in_keys = self.get_data_io_dict_keys(io_type)
        ns_manager = self.ee.ns_manager

        # original_data = deepcopy(dict_in)
        dict_out_keys = []
        for key in dict_in_keys:
            namespaced_key = ns_manager.get_namespaced_variable(
                self, key, io_type)
            dict_out_keys.append(namespaced_key)
        return dict_out_keys

    def _prepare_data_dict(self, io_type, data_dict=None):
        if data_dict is None:
            data_dict = self.get_data_io_dict(io_type)
        for key in data_dict.keys():
            curr_data = data_dict[key]
            data_keys = curr_data.keys()
            curr_data[self.IO_TYPE] = io_type
            curr_data[self.TYPE_METADATA] = None
            curr_data[self.VAR_NAME] = key
            if self.USER_LEVEL not in data_keys:
                curr_data[self.USER_LEVEL] = 1
            if self.RANGE not in data_keys:
                curr_data[self.RANGE] = None
            if self.UNIT not in data_keys:
                curr_data[self.UNIT] = None
            if self.DESCRIPTION not in data_keys:
                curr_data[self.DESCRIPTION] = None
            if self.POSSIBLE_VALUES not in data_keys:
                curr_data[self.POSSIBLE_VALUES] = None
            if data_dict[key]['type'] in ['array', 'dict', 'dataframe']:
                if self.DATAFRAME_DESCRIPTOR not in data_keys:
                    curr_data[self.DATAFRAME_DESCRIPTOR] = None
                if self.DATAFRAME_EDITION_LOCKED not in data_keys:
                    curr_data[self.DATAFRAME_EDITION_LOCKED] = True
            # For dataframes but also dict of dataframes...
            if data_dict[key]['type'] in ['dict', 'dataframe']:
                if self.DF_EXCLUDED_COLUMNS not in data_keys:
                    curr_data[self.DF_EXCLUDED_COLUMNS] = self.DEFAULT_EXCLUDED_COLUMNS

            if self.DISCIPLINES_FULL_PATH_LIST not in data_keys:
                curr_data[self.DISCIPLINES_FULL_PATH_LIST] = []
            if self.VISIBILITY not in data_keys:
                curr_data[self.VISIBILITY] = self.LOCAL_VISIBILITY
            if self.DEFAULT not in data_keys:
                if curr_data[self.VISIBILITY] == self.INTERNAL_VISIBILITY:
                    raise Exception(
                        f'The variable {key} in discipline {self.sos_name} must have a default value because its visibility is Internal')
                else:
                    curr_data[self.DEFAULT] = None
            else:
                curr_data[self.VALUE] = data_dict[key][self.DEFAULT]
            # -- Initialize VALUE to None by default
            if self.VALUE not in data_keys:
                curr_data[self.VALUE] = None
            if self.COUPLING not in data_keys:
                curr_data[self.COUPLING] = False
            if self.OPTIONAL not in data_keys:
                curr_data[self.OPTIONAL] = False
            if self.NUMERICAL not in data_keys:
                curr_data[self.NUMERICAL] = False
            if self.META_INPUT not in data_keys:
                curr_data[self.META_INPUT] = False

            # -- Outputs are not EDITABLE
            if self.EDITABLE not in data_keys:
                if curr_data[self.VISIBILITY] == self.INTERNAL_VISIBILITY:
                    curr_data[self.EDITABLE] = False
                elif self.CONNECTOR_DATA in curr_data.keys() and curr_data[self.CONNECTOR_DATA] is not None:
                    curr_data[self.EDITABLE] = False
                else:
                    curr_data[self.EDITABLE] = (io_type == self.IO_TYPE_IN)
            # -- Add NS_REFERENCE
            if curr_data[self.VISIBILITY] not in self.AVAILABLE_VISIBILITIES:
                var_name = curr_data[self.VAR_NAME]
                visibility = curr_data[self.VISIBILITY]
                raise ValueError(self.sos_name + '.' + var_name + ': visibility ' + str(
                    visibility) + ' not in allowed visibilities: ' + str(self.AVAILABLE_VISIBILITIES))
            if self.NAMESPACE in data_keys:
                curr_data[self.NS_REFERENCE] = self.get_ns_reference(
                    curr_data[self.VISIBILITY], curr_data[self.NAMESPACE])
            else:
                curr_data[self.NS_REFERENCE] = self.get_ns_reference(
                    curr_data[self.VISIBILITY])

            # store structuring variables in self._structuring_variables
            if self.STRUCTURING in data_keys and curr_data[self.STRUCTURING] is True:
                self._structuring_variables[key] = None
                del curr_data[self.STRUCTURING]

        return data_dict

    def get_sosdisc_inputs(self, keys=None, in_dict=False):
        """Accessor for the inputs values as a list

        :param data_names: the data names list
        :returns: the data list
        """

        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            keys = list(self.get_data_in().keys())
            in_dict = True
        inputs = self._get_sosdisc_io(keys, io_type=self.IO_TYPE_IN)

        if not in_dict:
            # return inputs in an ordered tuple (default)
            return inputs
        else:
            # return inputs in an dictionary
            formated_inputs = {}
            for key, val in zip(keys, inputs):
                formated_inputs[key] = val
            return formated_inputs

    def get_sosdisc_outputs(self, keys=None, in_dict=False):
        """Accessor for the outputs values as a list

        :param data_names: the data names list
        :returns: the data list
        """
        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            keys = [d[self.VAR_NAME] for d in self.get_data_out().values()]
            in_dict = True
        outputs = self._get_sosdisc_io(keys, io_type=self.IO_TYPE_OUT)
        if not in_dict:
            # return outputs in an ordered tuple (default)
            return outputs
        else:
            # return outputs in an dictionary
            formated_outputs = {}
            for key, val in zip(keys, outputs):
                formated_outputs[key] = val
            return formated_outputs

    def _get_sosdisc_io(self, keys, io_type):
        """ generic method to retrieve sos inputs and outputs
        """
        # convert local key names to namespaced ones
        variables = self._convert_list_of_keys_to_namespace_name(keys, io_type)
        if isinstance(variables, string_types):
            return self.dm.get_value(variables)
        elif isinstance(variables, list):
            values_list = [self.dm.get_value(
                key) for key in variables if key in self.dm.data_id_map]

            if len(values_list) != len(variables):
                missing_keys = [
                    key for key in variables if key not in self.dm.data_id_map]
                raise Exception(
                    f'The keys {missing_keys} for the discipline {self.get_disc_full_name()} are missing in the data manager')

            return values_list

    # -- execute/runtime section
    def execute(self, input_data=None):
        """
        Overwrite execute method from MDODiscipline to load input_data from datamanager if possible
        IMPORTANT NOTE: input_data should NOT be filled when execute called outside GEMS
        """
        if input_data is None:
            # if no input_data, i.e. not called by GEMS
            input_data = self.get_input_data_for_gems()
        else:
            # if input_data exists, i.e. called by GEMS
            # no need to convert the data
            pass

        # update status to 'PENDING' after configuration
        self.update_status_pending()

        result = None
        try:
            result = MDODiscipline.execute(self, input_data=input_data)
        except Exception as error:
            # Update data manager status (status 'FAILED' is not propagate correctly due to exception
            # so we have to force data manager status update in this case
            self._update_status_dm(self.status)
            raise error

        # When execution is done, is the status is again to 'pending' then we have to check if execution has been used
        # If execution cache is used, then the discipline is not run and its
        # status is not changed
        if (self.status == SoSDiscipline.STATUS_PENDING and self._cache_was_loaded is True):
            self._update_status_recursive(self.STATUS_DONE)

        self.__check_nan_in_data(result)

        return result

    def linearize(self, input_data=None, force_all=False, force_no_exec=False):
        """overloads GEMS linearize function
        """
        # set GEM's default_inputs for gradient computation purposes
        # to be deleted during GEMS update
        if input_data is not None:
            self.default_inputs = input_data
        else:
            self.default_inputs = {}
            input_data = self.get_input_data_for_gems()
            input_data = self._convert_float_into_array(input_data)
            self.default_inputs = input_data

        if self.linearization_mode == self.COMPLEX_STEP:
            # is complex_step, switch type of inputs variables
            # perturbed to complex
            inputs, _ = self._retreive_diff_inouts(force_all)
            def_inputs = self.default_inputs
            for name in inputs:
                def_inputs[name] = def_inputs[name].astype('complex128')
        else:
            pass

        if not force_no_exec:
            self.reset_statuses_for_run()
            self.exec_for_lin = True
            self.execute(input_data)
            self.exec_for_lin = False
            self.local_data = self._convert_float_into_array(self.local_data)
            force_no_exec = True

#         if not self._linearize_on_last_state:
#             # self.local_data.update(input_data)
#             input_data_sostrades = self._convert_array_into_new_type(
#                 input_data)
#             self.dm.set_values_from_dict(input_data_sostrades)

        if self.check_linearize_data_changes and not self.is_sos_coupling:
            disc_data_before_linearize = self.__get_discipline_inputs_outputs_dict_formatted__()

        result = MDODiscipline.linearize(
            self, input_data, force_all, force_no_exec)

        self.__check_nan_in_data(result)
        if self.check_linearize_data_changes and not self.is_sos_coupling:
            disc_data_after_linearize = self.__get_discipline_inputs_outputs_dict_formatted__()

            self.__check_discipline_data_integrity(disc_data_before_linearize,
                                                   disc_data_after_linearize,
                                                   'Discipline data integrity through linearize')

        return result

    def check_jacobian(self, input_data=None, derr_approx=MDODiscipline.FINITE_DIFFERENCES,
                       step=1e-7, threshold=1e-8, linearization_mode='auto',
                       inputs=None, outputs=None, parallel=False,
                       n_processes=MDODiscipline.N_CPUS,
                       use_threading=False, wait_time_between_fork=0,
                       auto_set_step=False, plot_result=False,
                       file_path="jacobian_errors.pdf",
                       show=False, figsize_x=10, figsize_y=10, input_column=None, output_column=None,
                       dump_jac_path=None, load_jac_path=None):
        """
        Overload check jacobian to execute the init_execution
        """

        self.init_execution()

        # if dump_jac_path is provided, we trigger GEMSEO dump
        if dump_jac_path is not None:
            reference_jacobian_path = dump_jac_path
            save_reference_jacobian = True
        # if dump_jac_path is provided, we trigger GEMSEO dump
        elif load_jac_path is not None:
            reference_jacobian_path = load_jac_path
            save_reference_jacobian = False
        else:
            reference_jacobian_path = None
            save_reference_jacobian = False

        approx = DisciplineJacApprox(
            self,
            derr_approx,
            step,
            parallel,
            n_processes,
            use_threading,
            wait_time_between_fork,
        )
        if inputs is None:
            inputs = self.get_input_data_names()
        if outputs is None:
            outputs = self.get_output_data_names()

        if auto_set_step:
            approx.auto_set_step(outputs, inputs, print_errors=True)

        # Differentiate analytically
        self.add_differentiated_inputs(inputs)
        self.add_differentiated_outputs(outputs)
        self.linearization_mode = linearization_mode
        self.reset_statuses_for_run()
        # Linearize performs execute() if needed
        self.linearize(input_data)

        if input_column is None and output_column is None:
            indices = None
        else:
            indices = self._get_columns_indices(
                inputs, outputs, input_column, output_column)

        o_k = approx.check_jacobian(
            self.jac,
            outputs,
            inputs,
            self,
            threshold,
            plot_result=plot_result,
            file_path=file_path,
            show=show,
            figsize_x=figsize_x,
            figsize_y=figsize_y,
            reference_jacobian_path=reference_jacobian_path,
            save_reference_jacobian=save_reference_jacobian,
            indices=indices,
        )
        return o_k

    def _get_columns_indices(self, inputs, outputs, input_column, output_column):
        """
        returns indices of input_columns and output_columns
        """
        # Get boundaries of the jacobian to compare
        if inputs is None:
            inputs = self.get_input_data_names()
        if outputs is None:
            outputs = self.get_output_data_names()

        indices = None
        if input_column is not None or output_column is not None:
            if len(inputs) == 1 and len(outputs) == 1:

                if self.sos_disciplines is not None:
                    for discipline in self.sos_disciplines:
                        self.jac_boundaries.update(discipline.jac_boundaries)

                indices = {}
                if output_column is not None:
                    jac_bnd = self.jac_boundaries[f'{outputs[0]},{output_column}']
                    tup = [jac_bnd['start'], jac_bnd['end']]
                    indices[outputs[0]] = [i for i in range(*tup)]

                if input_column is not None:
                    jac_bnd = self.jac_boundaries[f'{inputs[0]},{input_column}']
                    tup = [jac_bnd['start'], jac_bnd['end']]
                    indices[inputs[0]] = [i for i in range(*tup)]

            else:
                raise Exception(
                    'Not possible to use input_column and output_column options when \
                    there is more than one input and output')

        return indices

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Over load of the GEMS function 
        Compute the analytic jacobian of a discipline/model 
        Check if the jacobian in compute_sos_jacobian is OK 

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        if self.check_linearize_data_changes:
            disc_data_before_linearize = self.__get_discipline_inputs_outputs_dict_formatted__()
        # Initialize all matrices to zeros
        self._init_jacobian(inputs, outputs, with_zeros=True)

        self.compute_sos_jacobian()
        if self.check_linearize_data_changes:
            disc_data_after_linearize = self.__get_discipline_inputs_outputs_dict_formatted__()

            self.__check_discipline_data_integrity(disc_data_before_linearize,
                                                   disc_data_after_linearize,
                                                   'Discipline data integrity through compute_sos_jacobian')
        if self.check_min_max_gradients:
            self._check_min_max_gradients(self.jac)

    def _check_min_max_gradients(self, jac):
        '''Check of minimum and maximum jacobian values 
        '''

        for out in jac:
            for inp in self.jac[out]:
                grad = self.jac[out][inp]
                # avoid cases when gradient is not required
                if grad.size > 0:
                    d_name = self.get_disc_full_name()
                    #                     cond_number = np.linalg.cond(grad)
                    #                     if cond_number > 1e10 and not np.isinf(cond_number):
                    #                         self.logger.info(
                    # f'The Condition number of the jacobian dr {out} / dr {inp} is
                    # {cond_number}')
                    mini = np_min(grad)
                    if mini < -1e4:
                        self.logger.info(
                            "in discipline <%s> : dr<%s> / dr<%s>: minimum gradient value is <%s>" % (
                                d_name, out, inp, mini))

                    maxi = np_max(grad)
                    if maxi > 1e4:
                        self.logger.info(
                            "in discipline <%s> : dr<%s> / dr<%s>: maximum gradient value is <%s>" % (
                                d_name, out, inp, maxi))

    #                     grad_abs = np_abs(grad)
    #                     low_grad_ind = where(grad_abs < 1e-4)[0]
    #                     if low_grad_ind.size > 0 :
    #                         self.logger.info(
    #                             "in discipline <%s> : dr<%s> / dr<%s>: minimum abs gradient value is <%s>" % (d_name, out, inp, grad[low_grad_ind]))

    def compute_sos_jacobian(self):
        """Compute the analytic jacobian of a discipline/model 

        To be overloaded by sub classes.
        if we need to compute the jacobian this class MUST be implemented else it will return a zeros matrix
        """
        raise NotImplementedError(
            f'The discipline {self.get_disc_full_name()} has no compute_sos_jacobian function (if the jacobian is an empty matrix a pass is needed)')

    def set_partial_derivative(self, y_key, x_key, value):
        '''
        Set the derivative of y_key by x_key inside the jacobian of GEMS self.jac
        '''
        new_y_key = self.get_var_full_name(y_key, self._data_out)

        new_x_key = self.get_var_full_name(x_key, self._data_in)

        if new_x_key in self.jac[new_y_key]:
            self.jac[new_y_key][new_x_key] = value

    def set_partial_derivative_for_other_types(self, y_key_column, x_key_column, value):
        '''
        Set the derivative of the column y_key by the column x_key inside the jacobian of GEMS self.jac
        y_key_column = 'y_key,column_name'
        '''
        if len(y_key_column) == 2:
            y_key, y_column = y_key_column
        else:
            y_key = y_key_column[0]
            y_column = None

        lines_nb_y, index_y_column = self.get_boundary_jac_for_columns(
            y_key, y_column, self.IO_TYPE_OUT)

        if len(x_key_column) == 2:
            x_key, x_column = x_key_column
        else:
            x_key = x_key_column[0]
            x_column = None

        lines_nb_x, index_x_column = self.get_boundary_jac_for_columns(
            x_key, x_column, self.IO_TYPE_IN)

        # Convert keys in namespaced keys in the jacobian matrix for GEMS
        new_y_key = self.get_var_full_name(y_key, self._data_out)

        new_x_key = self.get_var_full_name(x_key, self._data_in)

        # Code when dataframes are filled line by line in GEMS, we keep the code for now
        #         if index_y_column and index_x_column is not None:
        #             for iy in range(value.shape[0]):
        #                 for ix in range(value.shape[1]):
        #                     self.jac[new_y_key][new_x_key][iy * column_nb_y + index_y_column,
        # ix * column_nb_x + index_x_column] = value[iy, ix]

        if new_x_key in self.jac[new_y_key]:
            if index_y_column is not None and index_x_column is not None:
                self.jac[new_y_key][new_x_key][index_y_column * lines_nb_y:(index_y_column + 1) * lines_nb_y,
                                               index_x_column * lines_nb_x:(index_x_column + 1) * lines_nb_x] = value
                self.jac_boundaries.update({f'{new_y_key},{y_column}': {'start': index_y_column * lines_nb_y,
                                                                        'end': (index_y_column + 1) * lines_nb_y},
                                            f'{new_x_key},{x_column}': {'start': index_x_column * lines_nb_x,
                                                                        'end': (index_x_column + 1) * lines_nb_x}})

            elif index_y_column is None and index_x_column is not None:
                self.jac[new_y_key][new_x_key][:, index_x_column *
                                               lines_nb_x:(index_x_column + 1) * lines_nb_x] = value

                self.jac_boundaries.update({f'{new_y_key},{y_column}': {'start': 0,
                                                                        'end': -1},
                                            f'{new_x_key},{x_column}': {'start': index_x_column * lines_nb_x,
                                                                        'end': (index_x_column + 1) * lines_nb_x}})
            elif index_y_column is not None and index_x_column is None:
                self.jac[new_y_key][new_x_key][index_y_column * lines_nb_y:(index_y_column + 1) * lines_nb_y,
                                               :] = value
                self.jac_boundaries.update({f'{new_y_key},{y_column}': {'start': index_y_column * lines_nb_y,
                                                                        'end': (index_y_column + 1) * lines_nb_y},
                                            f'{new_x_key},{x_column}': {'start': 0,
                                                                        'end': -1}})
            else:
                raise Exception(
                    'The type of a variable is not yet taken into account in set_partial_derivative_for_other_types')

    def get_boundary_jac_for_columns(self, key, column, io_type):

        data_io = self.get_data_io_dict(io_type)
        index_column = None
        key_type = data_io[key][self.TYPE]

        if key_type == 'dataframe':
            # Get the number of lines and the index of column from the metadata
            metadata = data_io[key][self.TYPE_METADATA][0]
            lines_nb = metadata['shape'][0]
            # delete the + 1 if we delete the index column
            index_column = metadata['columns'].index(column)
        elif key_type == 'array' or key_type == 'float':
            lines_nb = None
            index_column = None
        elif key_type == 'dict':
            value = data_io[key][self.VALUE]
            metadata = data_io[key][self.TYPE_METADATA]
            dict_keys = [meta['key'][0] for meta in metadata]
            lines_nb = len(value[column])
            index_column = dict_keys.index(column)

        return lines_nb, index_column

    def get_input_data_for_gems(self):
        '''
        Get input_data for linearize sosdiscipline
        '''
        input_data = {}
        input_data_names = self.input_grammar.get_data_names()
        if len(input_data_names) > 0:

            for data_name in input_data_names:
                input_data[data_name] = self.ee.dm.get_value(data_name)
            # convert sostrades types into numpy arrays
            # no need to update DM since call by SoSTrades
            input_data = self._convert_new_type_into_array(
                var_dict=input_data)

        return input_data

    def _run(self, update_local_data=True):
        ''' GEMS run method overloaded. Calls specific run() method from SoSDiscipline
        Defines the execution of the process, given that data has been checked.
        '''
        # Add an exception handler in order to have the capabilities to log
        # the exception before GEMS (when GEMS manage an error it does not propagate it and does
        # not record the stackstrace)
        try:
            # data conversion GEMS > SosStrades
            self._update_type_metadata()
            local_data_updt = self._convert_array_into_new_type(
                self.local_data)

            # update DM
            self.dm.set_values_from_dict(local_data_updt)

            # execute model
            self._update_status_dm(self.STATUS_RUNNING)

            if self.check_if_input_change_after_run and not self.is_sos_coupling:
                disc_inputs_before_execution = {self.get_var_full_name(key, self._data_in): {'value': value}
                                                for key, value in deepcopy(self.get_sosdisc_inputs()).items()}

            self.run()
            self.fill_output_value_connector()
            if self.check_if_input_change_after_run and not self.is_sos_coupling:
                disc_inputs_after_execution = {self.get_var_full_name(key, self._data_in): {'value': value}
                                               for key, value in deepcopy(self.get_sosdisc_inputs()).items()}
                self.__check_discipline_data_integrity(disc_inputs_before_execution,
                                                       disc_inputs_after_execution,
                                                       'Discipline inputs integrity through run')

        except Exception as exc:
            self._update_status_dm(self.STATUS_FAILED)
            self.logger.exception(exc)
            raise exc

        if update_local_data:
            out_dict = self._convert_coupling_outputs_into_gems_format()
            #-- Local data is the output dictionary for a GEMS discipline
            self.local_data.update(out_dict)  # update output data for gems

        # Make a test regarding discipline children status. With GEMS parallel execution, child discipline
        # can failed but FAILED (without an forward exception) status is not
        # correctly propagate upward
        if len(self.sos_disciplines) > 0:
            failed_list = list(filter(
                lambda d: d.status == self.STATUS_FAILED, self.sos_disciplines))

            if len(failed_list) > 0:
                raise SoSDisciplineException(
                    f'An exception occurs during execution in \'{self.name}\' discipline.')

        self._update_status_dm(self.STATUS_DONE)

    def _update_type_metadata(self):
        ''' update metadata of values not supported by GEMS
            (for cases where the data has been converted by the coupling)
        '''
        for var_name in self._data_in.keys():
            var_f_name = self.get_var_full_name(var_name, self._data_in)
            var_type = self.dm.get_data(var_f_name, self.VAR_TYPE_ID)
            if var_type in self.NEW_VAR_TYPE:
                if self.dm.get_data(var_f_name, self.TYPE_METADATA) is not None:
                    self._data_in[var_name][self.TYPE_METADATA] = self.dm.get_data(
                        var_f_name, self.TYPE_METADATA)

        for var_name in self._data_out.keys():
            var_f_name = self.get_var_full_name(var_name, self._data_out)
            var_type = self.dm.get_data(var_f_name, self.VAR_TYPE_ID)
            if var_type in self.NEW_VAR_TYPE:
                if self.dm.get_data(var_f_name, self.TYPE_METADATA) is not None:
                    self._data_out[var_name][self.TYPE_METADATA] = self.dm.get_data(
                        var_f_name, self.TYPE_METADATA)

    def run(self):
        ''' To be overloaded by sublcasses
        '''
        raise NotImplementedError()

    def _update_study_ns_in_varname(self, names):
        ''' updates the study name in the variable input names
        '''
        study = self.ee.study_name
        new_names = []
        for n in names:
            if not n.startswith(study):
                suffix = NS_SEP.join(n.split(NS_SEP)[1:])
                new_name = study + NS_SEP + suffix
            else:
                new_name = n
            new_names.append(new_name)
        return new_names

    def store_sos_outputs_values(self, dict_values):
        ''' store outputs from 'dict_values' into self._data_out
        '''
        # fill data using data connector if needed
        self._update_with_values(self._data_out, dict_values)

    def fill_output_value_connector(self):
        """
        get value of output variable with data connector
        """
        self._data_out
        updated_values = {}
        for key in self._data_out.keys():
            # if data connector is needed, use it
            if self.CONNECTOR_DATA in self._data_out[key].keys():
                if self._data_out[key][self.CONNECTOR_DATA] is not None:
                    # desc out is used because user update desc out keys.

                    updated_values[key] = ConnectorFactory.use_data_connector(
                        self._data_out[key][self.CONNECTOR_DATA],
                        self.ee.logger)

        self.store_sos_outputs_values(updated_values)

    def update_meta_data_out(self, new_data_dict):
        """
        update meta data of _data_out and DESC_OUT

        :param: new_data_dict, contains the data to be updated
        :type: dict
        :format: {'variable_name' : {'meta_data_name' : 'meta_data_value',...}....}
        """
        for key in new_data_dict.keys():
            for meta_data in new_data_dict[key].keys():
                self._data_out[key][meta_data] = new_data_dict[key][meta_data]
                if meta_data in self.DESC_OUT[key].keys():
                    self.DESC_OUT[key][meta_data] = new_data_dict[key][meta_data]

    def clean_dm_from_disc(self):

        self.dm.clean_from_disc(self.disc_id)

    def _set_dm_disc_info(self):
        disc_ns_name = self.get_disc_full_name()
        disc_dict_info = {}
        disc_dict_info['reference'] = self
        disc_dict_info['classname'] = self.__class__.__name__
        disc_dict_info['model_name'] = self.__module__.split('.')[-2]
        disc_dict_info['model_name_full_path'] = self.__module__
        disc_dict_info['treeview_order'] = 'no'
        disc_dict_info[self.NS_REFERENCE] = self.ee.ns_manager.get_local_namespace(
            self)
        self.disc_id = self.dm.update_disciplines_dict(
            self.disc_id, disc_dict_info, disc_ns_name)

    def get_var_full_name(self, var_name, disc_dict):
        ''' Get namespaced variable from namespace and var_name in disc_dict
        '''
        ns_reference = disc_dict[var_name][self.NS_REFERENCE]
        complete_var_name = disc_dict[var_name][self.VAR_NAME]
        var_f_name = self.ee.ns_manager.compose_ns(
            [ns_reference.value, complete_var_name])
        return var_f_name

    def update_from_dm(self):
        """
        Update all disciplines with datamanager information
        """

        for var_name in self._data_in.keys():
            var_f_name = self.get_var_full_name(var_name, self._data_in)

            default_val = self.dm.data_dict[self.dm.get_data_id(
                var_f_name)][self.DEFAULT]
            if self.dm.get_value(var_f_name) is None and default_val is not None:
                self._data_in[var_name][self.VALUE] = default_val
            else:
                # update from dm for all sos_disciplines to load all data
                self._data_in[var_name][self.VALUE] = self.dm.get_value(
                    var_f_name)
        # -- update sub-disciplines
        for discipline in self.sos_disciplines:
            discipline.update_from_dm()

    # -- Ids and namespace handling
    def get_disc_full_name(self):
        '''
        Returns the discipline name with full namespace
        '''
        return self.ee.ns_manager.get_local_namespace_value(self)

    def get_disc_id_from_namespace(self):

        return self.ee.dm.get_discipline_ids_list(self.get_disc_full_name())

    def _convert_list_of_keys_to_namespace_name(self, keys, io_type):

        # Refactor  variables keys with namespace
        if isinstance(keys, list):
            variables = [self._convert_to_namespace_name(
                key, io_type) for key in keys]
        else:
            variables = self._convert_to_namespace_name(
                keys, io_type)
        return variables

    def __filter_couplings_for_gems(self, io_type):
        ''' 
        Filter coupling before sending to GEMS 
        '''
        full_dict = self.get_data_io_dict(io_type)
        filtered_keys = []
        for var_name, value in full_dict.items():
            # Check if the param is a numerical parameter (function overload in
            # soscoupling)
            if self.delete_numerical_parameters_for_gems(
                    var_name):
                continue
            # Get the full var name
            full_var_name = self.get_var_full_name(
                var_name, self.get_data_io_dict(io_type))
            var_type_id = value[self.VAR_TYPE_ID]
            # if var type not covered by GEMS
            if var_type_id not in self.VAR_TYPE_GEMS:
                # if var type covered by available extended types
                if var_type_id in self.NEW_VAR_TYPE:
                    filtered_keys.append(full_var_name)
            else:
                filtered_keys.append(full_var_name)

        return filtered_keys

    def delete_numerical_parameters_for_gems(self, var_name):

        if var_name in self.NUM_DESC_IN:
            return True
        else:
            return False

    def _init_grammar_with_keys(self, names, io_type):
        ''' initialize GEMS grammar with names and type None
        '''
        names_dict = dict.fromkeys(names, None)
        if io_type == self.IO_TYPE_IN:
            grammar = self.input_grammar
            grammar.clear()

        elif io_type == self.IO_TYPE_OUT:
            grammar = self.output_grammar
            grammar.clear()
        grammar.initialize_from_base_dict(names_dict)

        return grammar

    def _convert_to_namespace_name(self, key, io_type):
        ''' Convert to namepsace with coupling_namespace management
            Using a key (variables name) and reference_data (yaml in or out),
            build the corresponding namespaced key using the visibility property
        '''

        # Refactor  variables keys with namespace
        result = self.ee.ns_manager.get_namespaced_variable(
            self, key, io_type)

        return result

    def _convert_coupling_outputs_into_gems_format(self):
        ''' convert discipline outputs, that could include data not
            handled by GEMS, into data handled by GEMS
        '''
        out_keys = self.get_output_data_names()
        # check out_keys types and convert NEW TYPE into GEMS TYPE
        out_dict = {}
        for var_f_name in out_keys:
            var_name = self.dm.get_data(var_f_name, self.VAR_NAME)
            out_dict[var_f_name] = self._data_out[var_name][self.VALUE]
        return self._convert_new_type_into_array(
            out_dict)

    # -- status handling section
    def _update_status_dm(self, status):

        # Avoid unnecessary call to status property (which can trigger event in
        # case of change)
        if self.status != status:
            self.status = status

        # Force update into discipline_dict (GEMS can change status but cannot update the
        # discipline_dict

        self.dm.disciplines_dict[self.disc_id]['status'] = status

    def update_status_pending(self):
        # keep reference branch status to 'REFERENCE'
        self._update_status_recursive(self.STATUS_PENDING)

    def update_status_running(self):
        # keep reference branch status to 'REFERENCE'
        self._update_status_recursive(self.STATUS_RUNNING)

    def _update_status_recursive(self, status):
        # keep reference branch status to 'REFERENCE'
        self._update_status_dm(status)
        for disc in self.sos_disciplines:
            disc._update_status_recursive(status)

    def _check_status_before_run(self):

        status_ok = True
        if self.status == self.STATUS_RUNNING:
            status_ok = False
        if self.re_exec_policy == self.RE_EXECUTE_NEVER_POLICY:
            if self.status not in [self.STATUS_PENDING,
                                   self.STATUS_CONFIGURE, self.STATUS_VIRTUAL]:
                status_ok = False
        elif self.re_exec_policy == self.RE_EXECUTE_DONE_POLICY:
            if self.status == self.STATUS_DONE:
                self.reset_statuses_for_run()
                status_ok = True
            elif self.status not in [self.STATUS_PENDING, self.STATUS_CONFIGURE, self.STATUS_VIRTUAL]:
                status_ok = False
        else:
            raise ValueError("Unknown re_exec_policy :" +
                             str(self.re_exec_policy))
        if not status_ok:
            raise ValueError("Trying to run a discipline " + str(type(self)) +
                             " with status: " + str(self.status) +
                             " while re_exec_policy is : " +
                             str(self.re_exec_policy))

    def _check_status(self, status):
        """
        Overload the gemseo.discipline _check_status method to take into
        account our new status (CONFIGURE) and let the gemseo.discipline class make
        its own assessment afterward

        :param status: the status to check
        :type status: string
        """
        if status != self.STATUS_CONFIGURE:
            super()._check_status(status)

    # -- Maturity handling section
    def set_maturity(self, maturity, maturity_dict=False):
        if maturity is None or maturity in self.possible_maturities or maturity_dict:
            self._maturity = maturity
        else:
            raise Exception(
                f'Unkown maturity {maturity} for discipline {self.sos_name}')

    def get_maturity(self):
        '''
        Get the maturity of the sosdiscipline, a discipline does not have any subdisciplines, only a coupling has
        '''
        if hasattr(self, '_maturity'):
            return self._maturity
        else:
            return ''

    def _build_dynamic_DESC_IN(self):
        pass

    def convert_string_to_int(self, val, val_data):
        '''
        Small function to convert a string into an int following the metadata known_values
        if the value is new, the int will be the len of the known values + 1
        '''
        if val not in val_data['known_values']:
            int_val = len(val_data['known_values']) + 1
            val_data['known_values'][val] = int_val
        else:
            int_val = val_data['known_values'][val]

        return int_val, val_data

    def _convert_dict_into_array(self, var_dict, values_list, metadata, prev_keys, prev_metadata):
        '''
        Convert a nested var_dict into a numpy array, and stores metadata
        useful to build the dictionary afterwards
        '''

        for key, val in var_dict.items():
            # for each value in the dictionary to convert
            nested_keys = prev_keys + [key]
            _type = type(val)
            # Previous metadata is used to get back the previous known values
            # for string to int conversion
            if prev_metadata is None:
                prev_metadata_key = None
            else:
                if len(prev_metadata) != 0.:
                    prev_metadata_key = prev_metadata.pop(0)
                else:
                    prev_metadata_key = None
            val_data = {}
            val_data['key'] = nested_keys
            val_data['type'] = _type
            if _type == dict:
                # if value is a nested dict
                metadata.append(val_data)
                values_list, metadata = self._convert_dict_into_array(
                    val, values_list, metadata, val_data['key'], prev_metadata)
            elif _type == DataFrame:
                # if value is a dataframe
                values_list, metadata = self._convert_df_into_array(
                    val, values_list, metadata, nested_keys)
            elif _type in self.STANDARD_TYPES:
                # if value is a int or float
                values_list = append(values_list, [val])
                metadata.append(val_data)
            elif _type == np_complex128:
                # for gradient analysis
                values_list = append(values_list, [val])
                val_data['type'] = np_float64
                metadata.append(val_data)
            elif _type in [list, ndarray]:
                # if val contains strings :
                if any(isinstance(elem, str) for elem in val):
                    val_data['known_values'] = {}
                    # We look for strings inside the list
                    for i_elem, elem in enumerate(val):
                        if isinstance(elem, str):
                            val_data_ielem = {}
                            val_data_ielem['known_values'] = {}
                            # when string is found we look for its known values
                            if prev_metadata_key is not None:
                                if i_elem < len(prev_metadata_key['known_values']) and 'known_values' in \
                                        prev_metadata_key['known_values'][i_elem]:
                                    val_data_ielem['known_values'] = prev_metadata_key['known_values'][i_elem][
                                        'known_values']
                            # convert the string into int and replace the
                            # string by this int in the list
                            int_val, val_data_ielem = self.convert_string_to_int(
                                elem, val_data_ielem)

                            val[i_elem] = int_val
                            val_data['known_values'][i_elem] = val_data_ielem

                if isinstance(val, list):
                    size = len(val)
                    val_data['shape'] = (size,)
                    val_data['size'] = size
                    values_list = append(values_list, val)
                else:
                    val_data['shape'] = val.shape
                    val_data['size'] = val.size
                    values_list = append(values_list, val.flatten())
                metadata.append(val_data)
            elif _type == str:
                # if value is a string look for is prev_metadata to find known
                # values

                if prev_metadata_key is not None and 'known_values' in prev_metadata_key:
                    val_data['known_values'] = prev_metadata_key['known_values']
                else:
                    val_data['known_values'] = {}
                # convert the string into int
                int_val, val_data = self.convert_string_to_int(
                    val, val_data)
                values_list = append(values_list, int_val)

                metadata.append(val_data)
            else:
                raise Exception(
                    f'The type {_type} in the dict {var_dict} is not taken into account')
        return values_list, metadata

    def _convert_df_into_array(
            self, var_df, values_list, metadata, keys, excluded_columns=DEFAULT_EXCLUDED_COLUMNS):
        ''' 
        Converts dataframe into array, and stores metada
        useful to build the dataframe afterwards
        '''
        # gather df data including index column
        #         data = var_df.to_numpy()

        val_data = {column: list(var_df[column].values)
                    for column in excluded_columns if column in var_df}

        new_var_df = var_df.drop(
            columns=[column for column in excluded_columns if column in var_df])

        val_data['indices'] = list(new_var_df.index.values)
        data = new_var_df.to_numpy()

        # indices = var_df.index.to_numpy()
        columns = new_var_df.columns.to_list()
        # To delete indices in convert delete the line below
        # data = hstack((atleast_2d(indices).T, values))

        val_data['key'] = keys
        val_data['type'] = DataFrame
        val_data['columns'] = columns
        val_data['shape'] = data.shape
        val_data['size'] = data.size
        val_data['dtypes'] = [new_var_df[col].dtype for col in columns]
        # to flatten by lines erase the option 'F' or put the 'C' option
        values_list = append(values_list, data.flatten(order='F'))
        metadata.append(val_data)
        return values_list, metadata

    def _convert_float_into_array(
            self, var_dict):
        ''' 
        Check element type in var_dict, convert float or int into numpy array
            in order to deal with linearize issues in GEMS
        '''
        for key, var in var_dict.items():
            if isinstance(var, (float, int, complex)):
                var_dict[key] = array([var])

        return var_dict

    def _convert_new_type_into_array(
            self, var_dict):
        ''' 
        Check element type in var_dict, convert new type into numpy array
            and stores metadata into DM for afterwards reconversion
        '''
        for key, var in var_dict.items():
            var_type = self.dm.get_data(key, self.VAR_TYPE_ID)
            if var_type in self.NEW_VAR_TYPE:
                if not isinstance(
                        var, self.VAR_TYPE_MAP[var_type]) and var is not None:
                    msg = f"Variable {key} has type {type(var)}, "
                    msg += f"however type {self.VAR_TYPE_MAP[var_type]} was expected."
                    msg += f'before run of discipline {self} with name {self.get_disc_full_name()} '
                    raise ValueError(msg)
                else:
                    if var is None:
                        var_dict[key] = None
                    else:
                        values_list = []
                        metadata = []
                        prev_key = []
                        if var_type in ['dict', 'string', 'string_list']:
                            prev_metadata = self.dm.get_data(
                                key, self.TYPE_METADATA)
                        # if type needs to be converted
                        if var_type == 'dict':
                            # if value is a dictionary
                            all_values = list(var.values())
                            if all([self._is_value_type_handled(val)
                                    for val in all_values]):
                                # convert if all values are handled by
                                # SoSTrades
                                values_list, metadata = self._convert_dict_into_array(
                                    var, values_list, metadata, prev_key, deepcopy(prev_metadata))
                            else:
                                evaluated_types = [type(val)
                                                   for val in all_values]
                                msg = f"\n Invalid type of parameter {key}: {var}/'{evaluated_types}' in discipline {self.sos_name}."
                                msg += f"\n Dictionary values must be among {list(self.VAR_TYPE_MAP.keys())}"
                                raise SoSDisciplineException(msg)
                        elif var_type == 'dataframe':
                            # if value is a DataFrame
                            excluded_columns = self.dm.get_data(
                                key, self.DF_EXCLUDED_COLUMNS)
                            values_list, metadata = self._convert_df_into_array(
                                var, values_list, metadata, prev_key, excluded_columns)
                        elif var_type == 'string':
                            # if value is a string
                            metadata_dict = {}
                            metadata_dict['known_values'] = {}
                            if prev_metadata is not None and 'known_values' in prev_metadata[0]:
                                metadata_dict['known_values'] = prev_metadata[0]['known_values']

                            values_list, metadata_dict = self.convert_string_to_int(
                                var, metadata_dict)

                            metadata.append(metadata_dict)

                        elif var_type == 'string_list':
                            # if value is a list of strings
                            for i_elem, elem in enumerate(var):
                                metadata_dict_elem = {}
                                metadata_dict_elem['known_values'] = {}
                                if prev_metadata is not None and i_elem < len(prev_metadata) and 'known_values' in \
                                        prev_metadata[i_elem]:
                                    metadata_dict_elem['known_values'] = prev_metadata[i_elem]['known_values']

                                value_elem, metadata_dict_elem = self.convert_string_to_int(
                                    elem, metadata_dict_elem)
                                values_list.append(value_elem)
                                metadata.append(metadata_dict_elem)
                        elif var_type in ['float', 'int']:
                            # store float into array for gems
                            metadata = {'var_type': type(var)}
                            values_list = array([var])

                        # update current dictionary value
                        var_dict[key] = values_list
                        # Update metadata
                        self.dm.set_data(key, self.TYPE_METADATA, metadata)

        return var_dict

    def _is_value_type_handled(self, val):
        return isinstance(val, tuple(self.VAR_TYPE_MAP.values())
                          ) or isinstance(val, np_complex128)

    # -- coupling variables handling
    def _convert_array_into_dict(self, arr_to_convert, new_data, val_datalist):
        # convert list into dict using keys from dm.data_dict
        if len(val_datalist) == 0:
            # means the dictionary is empty or None
            return {}
        else:
            metadata = val_datalist.pop(0)

        _type = metadata['type']
        _keys = metadata['key']

        nested_keys = _keys[:-1]
        to_update = self.__get_nested_val(new_data, nested_keys)
        _key = _keys[-1]
        # dictionaries
        if _type == dict:
            to_update[_key] = {}
            self._convert_array_into_dict(
                arr_to_convert, new_data, val_datalist)
        # DataFrames
        elif _type == DataFrame:
            _df = self._convert_array_into_df(arr_to_convert, metadata)
            to_update[_key] = _df
            _size = metadata['size']
            arr_to_convert = delete(arr_to_convert, arange(_size))
            if len(val_datalist) > 0:
                self._convert_array_into_dict(
                    arr_to_convert, new_data, val_datalist)
        # int, float, or complex
        elif _type in [int, float, np_int32, np_int64, np_float64, np_complex128, bool]:
            _val = arr_to_convert[0]
            arr_to_convert = delete(arr_to_convert, [0])
            to_update[_key] = _type(_val)
            if len(val_datalist) > 0:
                self._convert_array_into_dict(
                    arr_to_convert, new_data, val_datalist)
        # numpy array or list
        elif _type in [list, ndarray]:
            _shape = metadata['shape']
            _size = metadata['size']
            _arr = arr_to_convert[:_size]
            _arr = _arr.reshape(_shape)
            if _type == list:
                _arr = _arr.tolist()
            if 'known_values' in metadata:
                # Means that we have a string somewhere in the list or array
                for index_arr, metadata_ind in metadata['known_values'].items():
                    int_value = int(_arr[index_arr])
                    _arr[index_arr] = next((strg for strg, int_to_convert in metadata_ind['known_values'].items(
                    ) if int_to_convert == int_value), None)

            arr_to_convert = delete(arr_to_convert, arange(_size))

            to_update[_key] = _arr
            if len(val_datalist) > 0:
                self._convert_array_into_dict(
                    arr_to_convert, new_data, val_datalist)
        elif _type == str:
            to_convert = arr_to_convert[0]
            arr_to_convert = delete(arr_to_convert, [0])
            _val = next((strg for strg, int_to_convert in metadata['known_values'].items(
            ) if int_to_convert == to_convert), None)
            to_update[_key] = _type(_val)
            if len(val_datalist) > 0:
                self._convert_array_into_dict(
                    arr_to_convert, new_data, val_datalist)
        else:
            raise Exception(
                f'The type {_type} in the dict {arr_to_convert} is not taken into account')
        return to_update

    def _convert_array_into_df(self, arr_to_convert, metadata, excluded_columns=DEFAULT_EXCLUDED_COLUMNS):
        # convert list into dataframe using columns from dm.data_dict
        _shape = metadata['shape']
        _size = metadata['size']
        _col = metadata['columns']
        _dtypes = metadata['dtypes']
        _arr = arr_to_convert[:_size]
        # to flatten by lines erase the option 'F' or put the 'C' option
        import numpy as np
        if len(_arr) != np.prod(list(_shape)):
            print('wrong')
        # TO DO: test object type properly
        _arr = _arr.reshape(_shape, order='F')

        # indices = array([_arr[i, 0]
        #                 for i in range(len(_arr))]).real.astype(int64)

        df = DataFrame(data=_arr, columns=_col)

        for col, dtype in zip(_col, _dtypes):
            if len(df[col].values) > 0:
                if type(df[col].values[0]).__name__ != 'complex' and type(df[col].values[0]).__name__ != 'complex128':
                    df[col] = df[col].astype(dtype)

        if 'indices' in metadata:
            df.index = metadata['indices']

        for column_excl in excluded_columns:
            if column_excl in metadata:
                df.insert(loc=0, column=column_excl,
                          value=metadata[column_excl])

        return df

    def _convert_array_into_new_type(self, local_data):
        ''' convert list in local_data into correct type in data_in
            returns an updated copy of local_data
        '''
        local_data_updt = deepcopy(local_data)

        for key, to_convert in local_data_updt.items():
            # get value in DataManager
            _type = self.dm.get_data(
                key, self.VAR_TYPE_ID)
            metadata_list = self.dm.get_data(key, self.TYPE_METADATA)
            if to_convert is None:
                local_data_updt[key] = None
            else:
                # check dict type in data_to_update and visibility
                if _type == 'dict' or _type == 'df_dict':
                    if metadata_list is None:
                        raise ValueError(
                            "Variable %s in %s cannot be converted since no metadata is available." %
                            (key, self.get_disc_full_name()))
                    new_data = {}

                    local_data_updt[key] = self._convert_array_into_dict(
                        to_convert, new_data, deepcopy(metadata_list))

                # check dataframe type in data_in and visibility
                elif _type == 'dataframe':
                    if metadata_list is None:
                        raise ValueError(
                            "Variable %s in %s cannot be converted since no metadata is available." %
                            (key, self.get_disc_full_name()))
                    metadata = metadata_list[0]
                    excluded_columns = self.dm.get_data(
                        key, self.DF_EXCLUDED_COLUMNS)
                    local_data_updt[key] = self._convert_array_into_df(
                        to_convert, metadata, excluded_columns)
                elif _type == 'string':
                    metadata = metadata_list[0]

                    local_data_updt[key] = next((strg for strg, int_to_convert in metadata['known_values'].items(
                    ) if int_to_convert == to_convert), None)
                elif _type == 'string_list':
                    local_data_updt[key] = []
                    for i, val in enumerate(to_convert):
                        metadata = metadata_list[i]
                        local_data_updt[key].append(
                            next((strg for strg, int_to_convert in metadata['known_values'].items(
                            ) if int_to_convert == val), None))
                elif _type in ['float', 'int']:
                    if isinstance(to_convert, ndarray):
                        # Check if metadata has been created
                        # Check if the value is complex that means that we are
                        # in a complex step method do not kill the complex part
                        if metadata_list is not None and not isinstance(to_convert[0], complex):
                            # if both conditions are OK reuse the float type in
                            # the metadata for the value
                            local_data_updt[key] = metadata_list['var_type'](
                                to_convert[0])

                        else:
                            local_data_updt[key] = to_convert[0]
        return local_data_updt

    def __get_nested_val(self, dict_in, keys):
        ''' returns the value of a nested dictionary of depth len(keys)
        output : d[keys[0]][..][keys[n]]
        '''

        def func_dic(dict_in, key): return dict_in[key]

        nested_val = reduce(func_dic, keys, dict_in)

        return nested_val

    def get_chart_filter_list(self):
        """ Return a list of ChartFilter instance base on the inherited 
        class post processing filtering capabilities

        :return: ChartFilter[]
        """
        return []

    def get_post_processing_list(self, filters=None):
        """ Return a list of post processing instance using the ChartFilter list given
        as parameter

        :params: chart_fiters : filter to apply during post processing making
        :type: ChartFilter[]

        :return post processing instance list
        """

        return []

    def _init_shared_attrs(self):
        """Initialize the shared attributes in multiprocessing.

            This method is overriden because in case of huge amount of discipline
            pre reserved semaphores (3 per class) in MDODiscipline class cause an out of memory
            caused by the number of file descriptor allocated for those semaphore

            A mock class is used to provide semaphore methods
        """

        self._n_calls = SemMock(0)
        self._exec_time = SemMock(0)
        self._n_calls_linearize = SemMock(0)

    def set_configure_status(self, is_configured):
        """Set boolean is_configured which indicates if the discipline has been configured 
            to avoid several configuration in a multi-level process and save time """

        self._is_configured = is_configured

    def get_configure_status(self):
        """Get boolean is_configured which indicates if the discipline has been configured 
            to avoid several configuration in a multi-level process and save time """

        if hasattr(self, '_is_configured'):
            return self._is_configured
        else:
            return ''

    def is_configured(self):
        '''
        Return False if discipline needs to be configured, True if not
        '''
        return self.get_configure_status() and not self.check_structuring_variables_changes()

    def check_structuring_variables_changes(self):
        '''
        Compare structuring variables stored in discipline with values in dm
        Return True if at least one structuring variable value has changed, False if not
        '''
        dict_values_dm = {key: self.get_sosdisc_inputs(
            key) for key in self._structuring_variables.keys()}
        try:
            return dict_values_dm != self._structuring_variables
        except:
            diff_dict = {}
            compare_dict(dict_values_dm,
                         self._structuring_variables, '', diff_dict, df_equals=True)
            return diff_dict != {}

    def set_structuring_variables_values(self):
        '''
        Store structuring variables values from dm in self._structuring_variables
        '''
        for struct_var in list(self._structuring_variables.keys()):
            if struct_var in self._data_in:
                self._structuring_variables[struct_var] = deepcopy(
                    self.get_sosdisc_inputs(struct_var))

    # ----------------------------------------------------
    # ----------------------------------------------------
    #  METHODS TO DEBUG DISCIPLINE
    # ----------------------------------------------------
    # ----------------------------------------------------

    def __check_nan_in_data(self, data):
        """ Using entry data, check if nan value exist in data's

        :params: data
        :type: composite data

        """

        if self.nan_check:
            self.__check_nan_in_data_rec(data, "")

    def __check_nan_in_data_rec(self, data, parent_key):
        """ Using entry data, check if nan value exist in data's as recursive
        method

        :params: data
        :type: composite data

        :params: parent_key, on composite type (dict), reference parent key
        :type: str

        """
        import pandas as pd
        for data_key, data_value in data.items():

            nan_found = False
            if isinstance(data_value, DataFrame):
                if data_value.isnull().any():
                    nan_found = True
            elif isinstance(data_value, ndarray):
                # None value in list throw an exception when used with isnan
                if sum(1 for _ in filter(None.__ne__, data_value)) != len(data_value):
                    nan_found = True
                elif pd.isnull(list(data_value)).any():
                    nan_found = True
            elif isinstance(data_value, list):
                # None value in list throw an exception when used with isnan
                if sum(1 for _ in filter(None.__ne__, data_value)) != len(data_value):
                    nan_found = True
                elif pd.isnull(data_value).any():
                    nan_found = True
            elif isinstance(data_value, dict):
                self.__check_nan_in_data_rec(
                    data_value, f'{parent_key}/{data_key}')
            elif isinstance(data_value, floating):
                if pd.isnull(data_value).any():
                    nan_found = True

            if nan_found:
                full_key = data_key
                if len(parent_key) > 0:
                    full_key = f'{parent_key}/{data_key}'
                self.logger.debug(f'NaN values found in {full_key}')
                self.logger.debug(data_value)

    def __check_discipline_data_integrity(self, left_dict, right_dict, test_subject):
        from sos_trades_core.sos_processes.compare_data_manager_tooling import compare_dict

        dict_error = {}
        compare_dict(left_dict, right_dict, '', dict_error)

        if dict_error != {}:
            for error in dict_error:
                output_error = '\n'
                output_error += f'Error while test {test_subject} on sos discipline {self.sos_name} :\n'
                output_error += f'Mismatch in {error}: {dict_error.get(error)}'
                output_error += '\n---------------------------------------------------------'
                print(output_error)

    def __get_discipline_inputs_outputs_dict_formatted__(self):
        disc_inputs = {self.get_var_full_name(key, self._data_in): {'value': value}
                       for key, value in deepcopy(self.get_sosdisc_inputs()).items()}
        disc_outputs = {self.get_var_full_name(key, self._data_out): {'value': value}
                        for key, value in deepcopy(self.get_sosdisc_outputs()).items()}
        disc_data = {}
        disc_data.update(disc_inputs)
        disc_data.update(disc_outputs)

        return disc_data

    def get_infos_gradient(self, output_var_list, input_var_list):
        """ Method to linearize an sos_discipline object and get gradient of output_var_list wrt input_var_list

        :params: input_var_list
        :type: list

        :params: output_var_list
        :type: list        
        """

        dict_infos_values = {}
        self.add_differentiated_inputs(input_var_list)
        self.add_differentiated_outputs(output_var_list)
        result_linearize_dict = self.linearize()

        for out_var_name in output_var_list:
            dict_infos_values[out_var_name] = {}
            for in_var_name in input_var_list:
                dict_infos_values[out_var_name][in_var_name] = {}
                grad_value = result_linearize_dict[out_var_name][in_var_name]
                dict_infos_values[out_var_name][in_var_name]['min'] = grad_value.min(
                )
                dict_infos_values[out_var_name][in_var_name]['max'] = grad_value.max(
                )
                dict_infos_values[out_var_name][in_var_name]['mean'] = grad_value.mean(
                )

        return dict_infos_values

    def clean(self):
        """This method cleans a sos_discipline;
        In the case of a "simple" discipline, it removes the discipline from
        its father builder and from the factory sos_discipline. This is achieved
        by the method remove_sos_discipline of the factory
        """
        self.father_builder.remove_discipline(self)
        self.clean_dm_from_disc()
        self.ee.ns_manager.remove_dependencies_after_disc_deletion(
            self, self.disc_id)
        self.ee.factory.remove_sos_discipline(self)
