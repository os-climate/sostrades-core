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


class ProxyDisciplineException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'

# def _proxy_run(cls):
#     '''
#     uses user wrapp run during execution
#     '''
#     return cls.proxy_discipline.run()
# 
#     
# def _proxy_compute_jacobian(cls):
#     '''
#     uses user wrapp jacobian computation during execution
#     '''
#     return cls.proxy_discipline.compute_sos_jacobian()


class ProxyDiscipline(object):
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
        self.mdo_discipline = None
        self.sub_mdo_disciplines = []
        self.proxy_disciplines = []
        self.status = None

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
        self.built_proxy_disciplines = []
        self.in_checkjac = False
        self._is_configured = False
        self._set_children_cache_inputs = False

        # -- disciplinary data attributes
        self.inst_desc_in = None  # desc_in of instance used to add dynamic inputs
        self.inst_desc_out = None  # desc_out of instance used to add dynamic outputs
        self._data_in = None
        self._data_out = None
        self._structuring_variables = None
        self.reset_data()
        # -- Maturity attribute
        self._maturity = self.get_maturity()

        # Add the discipline in the dm and get its unique disc_id (was in the
        # configure)
        self._set_dm_disc_info()

        # update discipline status to CONFIGURE
        self._update_status_dm(self.STATUS_CONFIGURE)
        
    def _proxy_run(self):
        '''
        uses user wrapp run during execution
        '''
        return self.run()
        
    def _proxy_compute_jacobian(self):
        '''
        uses user wrapp jacobian computation during execution
        '''
        return self.compute_sos_jacobian()

    def prepare_execution(self):
            
        self.init_gemseo_discipline()
#         self.set_cache() -> TODO: be able to comment this line, by passing the cache_type option directly as MDODiscipline input
    
    def init_gemseo_discipline(self):
        '''
        Initialization of GEMSEO MDODisciplines
        To be overloaded by subclasses
        '''
        if self.mdo_discipline is None:
            disc = MDODiscipline(name=self.get_disc_full_name(),
                             grammar_type=self.SOS_GRAMMAR_TYPE,
                             cache_type=self.get_sosdisc_inputs(self.CACHE_TYPE))
            disc.proxy_discipline = self
            setattr(disc, '_run', self._proxy_run)
            setattr(disc, 'compute_sos_jacobian', self._proxy_compute_jacobian)
            self.mdo_discipline = disc
        
            disc._ATTR_TO_SERIALIZE += ("proxy_discipline",)
        
        self.update_gemseo_grammar_with_data_io()
        
    def update_gemseo_grammar_with_data_io(self):
        # Remove unavailable GEMS type variables before initialize
        # input_grammar
        if not self.is_sos_coupling:
            data_in = self.get_data_io_with_full_name(
                self.IO_TYPE_IN)
            data_out = self.get_data_io_with_full_name(
                self.IO_TYPE_OUT)
            self._init_grammar_with_keys(data_in, self.IO_TYPE_IN)
            self._init_grammar_with_keys(data_out, self.IO_TYPE_OUT)

    def _init_grammar_with_keys(self, names, io_type):
        ''' initialize GEMS grammar with names and type None
        '''
        names_dict = dict.fromkeys(names, None)
        disc = self.mdo_discipline
        if io_type == self.IO_TYPE_IN:
            grammar = disc.input_grammar
            grammar.clear()

        elif io_type == self.IO_TYPE_OUT:
            grammar = disc.output_grammar
            grammar.clear()
        grammar.initialize_from_base_dict(names_dict)

        return grammar

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
            
    def get_input_data_names(self):
        ''' returns the list of input data names,
        based on i/o and namespaces declarations in the user wrapper
        '''
        return list(self.get_data_io_with_full_name(self.IO_TYPE_IN).keys())
        
    def get_output_data_names(self):
        ''' returns the list of input data names,
        based on i/o and namespaces declarations in the user wrapper
        '''
        return list(self.get_data_io_with_full_name(self.IO_TYPE_OUT).keys())

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

#     def update_gems_grammar_with_data_io(self):
#         # Remove unavailable GEMS type variables before initialize
#         # input_grammar
#         if not self.is_sos_coupling:
#             data_in = self.get_data_io_with_full_name(
#                 self.IO_TYPE_IN)
#             data_out = self.get_data_io_with_full_name(
#                 self.IO_TYPE_OUT)
#             self.init_gems_grammar(data_in, self.IO_TYPE_IN)
#             self.init_gems_grammar(data_out, self.IO_TYPE_OUT)

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

#     def init_gems_grammar(self, data_keys, io_type):
#         '''
#         Init Gems grammar with keys from a data_in/out dict
#         io_type specifies 'IN' or 'OUT'
#         '''
#         self._init_grammar_with_keys(data_keys, io_type)

    def local_data(self):
        '''
         Property to obtain the local data of the mdo_discipline
        '''
        return self.mdo_discipline.local_data

    def set_local_data(self, local_data_value):
        '''
         Property to set the local data of the proxy discipline
        '''
        self.local_data = local_data_value

    def get_built_disciplines_ids(self):
        return [disc.name for disc in self.proxy_disciplines]

    def get_proxy_disciplines(self):

        return self.proxy_disciplines

    def get_sub_proxy_disciplines(self, disc_list=None):
        ''' recursively returns all subdisciplines
        '''
        if disc_list is None:
            disc_list = []
        for disc in self.proxy_disciplines:
            disc_list.append(disc)
            disc.get_sub_proxy_disciplines(disc_list)
        return disc_list

    @property
    def ordered_disc_list(self):
        '''
         Property to obtain the ordered list of disciplines by default for
         a sos_discipline it is the order of proxy_disciplines
        '''
        return self.proxy_disciplines

    def add_discipline(self, disc):
        ''' add a discipline
        '''
        self.proxy_disciplines.append(disc)
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

#         self.set_numerical_parameters()

        if self.check_structuring_variables_changes():
            self.set_structuring_variables_values()

        self.setup_sos_disciplines()

        self.reload_io()

        # update discipline status to CONFIGURE
        self._update_status_dm(self.STATUS_CONFIGURE)

        self.set_configure_status(True)

    def set_numerical_parameters(self):
        '''
        Set numerical parameters of the sos_discipline defined in the NUM_DESC_IN
        '''
        if self._data_in != {}:
            self.linearization_mode = self.get_sosdisc_inputs(
                'linearization_mode')
            cache_type = self.get_sosdisc_inputs(self.CACHE_TYPE)
            cache_file_path = self.get_sosdisc_inputs(self.CACHE_FILE_PATH)

            if cache_type != self._structuring_variables[self.CACHE_TYPE]:
                self._set_children_cache_inputs = True
                self.set_cache(self, cache_type, cache_file_path)

            # Debug mode
            debug_mode = self.get_sosdisc_inputs('debug_mode')
            if debug_mode == "nan":
                self.nan_check = True
            elif debug_mode == "input_change":
                self.check_if_input_change_after_run = True
            elif debug_mode == "linearize_data_change":
                self.check_linearize_data_changes = True
            elif debug_mode == "min_max_grad":
                self.check_min_max_gradients = True
            elif debug_mode == "min_max_couplings":
                self.check_min_max_couplings = True
            elif debug_mode == "all":
                self.nan_check = True
                self.check_if_input_change_after_run = True
                self.check_linearize_data_changes = True
                self.check_min_max_gradients = True
                self.check_min_max_couplings = True
            if debug_mode != "":
                if debug_mode == "all":
                    for mode in self.AVAILABLE_DEBUG_MODE:
                        if mode not in ["", "all"]:
                            self.logger.info(
                                f'Discipline {self.sos_name} set to debug mode {mode}')
                else:
                    self.logger.info(
                        f'Discipline {self.sos_name} set to debug mode {debug_mode}')
                    
#     def set_cache(self, disc, cache_type, cache_hdf_file):
#         '''
#         Instantiate and set cache for disc if cache_type is not 'None'
#         '''
#         if cache_type == MDOChain.HDF5_CACHE and cache_hdf_file is None:
#             raise Exception(
#                 'if the cache type is set to HDF5Cache, the cache_file path must be set')
#         else:
#             disc.cache = None
#             if cache_type != 'None':
#                 disc.set_cache_policy(
#                     cache_type=cache_type, cache_hdf_file=cache_hdf_file)
                
    def set_children_cache_inputs(self):
        '''
        Set cache_type and cache_file_path input values to children, if cache inputs have changed
        '''
        if self._set_children_cache_inputs:
            cache_type = self.get_sosdisc_inputs(ProxyDiscipline.CACHE_TYPE)
            cache_file_path = self.get_sosdisc_inputs(ProxyDiscipline.CACHE_FILE_PATH)
            for disc in self.proxy_disciplines:
                if ProxyDiscipline.CACHE_TYPE in disc._data_in:
                    self.dm.set_data(disc.get_var_full_name(
                        ProxyDiscipline.CACHE_TYPE, disc._data_in), self.VALUE, cache_type, check_value=False)
                    if cache_file_path is not None:
                        self.dm.set_data(disc.get_var_full_name(
                            ProxyDiscipline.CACHE_FILE_PATH, disc._data_in), self.VALUE, cache_file_path, check_value=False)
            self._set_children_cache_inputs = False

    def setup_sos_disciplines(self):
        '''
        Method to be overloaded to add dynamic inputs/outputs using add_inputs/add_outputs methods.
        If the value of an input X determines dynamic inputs/outputs generation, then the input X is structuring and the item 'structuring':True is needed in the DESC_IN
        DESC_IN = {'X': {'structuring':True}}
        '''
        pass

    def set_dynamic_default_values(self, default_values_dict):
        '''
        Method to set default value to a variable with short_name in a discipline when the default value varies with other input values
        i.e. a default array length depends on a number of years
        PARAM IN : default_values_dict : dict with key is variable short name and value is the default value
        '''

        for short_key, default_value in default_values_dict.items():
            if short_key in self._data_in:
                ns_key = self.get_var_full_name(short_key, self._data_in)
                self.dm.no_check_default_variables.append(ns_key)
                self.dm.set_data(ns_key, self.DEFAULT, default_value, False)
            else:
                self.logger.info(
                    f'Try to set a default value for the variable {short_key} in {self.sos_name} which is not an input of this discipline ')

    # -- cache handling

    def clear_cache(self):
        # -- Need to clear cache for gradients analysis
        if self.cache is not None:
            self.cache.clear()
        for discipline in self.proxy_disciplines:
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

    def get_data_io_with_full_name(self, io_type):
        data_io_short_name = self.get_data_io_dict(io_type)
        data_io_full_name = {self.get_var_full_name(
            var_name, data_io_short_name): value_dict for var_name, value_dict in data_io_short_name.items()}

        return data_io_full_name

    def get_data_with_full_name(self, io_type, full_name, data_name=None):

        data_io_full_name = self.get_data_io_with_full_name(io_type)

        if data_name is None:
            return data_io_full_name[full_name]
        else:
            return data_io_full_name[full_name][data_name]

    def _update_with_values(self, to_update, update_with, update_dm=False):
        ''' update <to_update> 'value' field with <update_with>
        '''
        to_update_local_data = {}
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
                    if to_update[key][self.TYPE] in self.UNSUPPORTED_GEMSEO_TYPES:
                        # data with unsupported types for gemseo
                        to_update_dm[self.get_var_full_name(
                            key, to_update)] = ns_update_with[key]
                    else:
                        to_update_local_data[self.get_var_full_name(
                            key, to_update)] = ns_update_with[key]

        if update_dm:
            # update DM after run
            self.dm.set_values_from_dict(to_update_local_data)
        else:
            # update local_data after run
            self.mdo_discipline.local_data.update(to_update_local_data)

        # need to update outputs that will disappear after filtering the
        # local_data with supported types
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

    def get_sosdisc_inputs(self, keys=None, in_dict=False, full_name=False):
        """Accessor for the inputs values as a list or dict

        :param keys: the input short names list
        :param in_dict: if output format is dict
        :param full_name: if keys in output are full names
        :returns: the inputs values list or dict
        """

        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            keys = list(self.get_data_in().keys())
            in_dict = True
        inputs = self._get_sosdisc_io(
            keys, io_type=self.IO_TYPE_IN, full_name=full_name)
        if in_dict:
            # return inputs in an dictionary
            return inputs
        else:
            # return inputs in an ordered tuple (default)
            if len(inputs) > 1:
                return list(inputs.values())
            else:
                return list(inputs.values())[0]

    def get_sosdisc_outputs(self, keys=None, in_dict=False, full_name=False):
        """Accessor for the outputs values as a list or dict

        :param keys: the output short names list
        :param in_dict: if output format is dict
        :param full_name: if keys in output are full names
        :returns: the outputs values list or dict
        """
        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            keys = [d[self.VAR_NAME] for d in self.get_data_out().values()]
            in_dict = True
        outputs = self._get_sosdisc_io(
            keys, io_type=self.IO_TYPE_OUT, full_name=full_name)
        if in_dict:
            # return outputs in an dictionary
            return outputs
        else:
            # return outputs in an ordered tuple (default)
            if len(outputs) > 1:
                return list(outputs.values())
            else:
                return list(outputs.values())[0]

    def _get_sosdisc_io(self, keys, io_type, full_name=False):
        """ Generic method to retrieve sos inputs and outputs

        :param keys: the data names list
        :param io_type: 'in' or 'out'
        :param full_name: if keys in returned dict are full names
        :returns: dict of keys values
        """

        # convert local key names to namespaced ones
        if isinstance(keys, str):
            keys = [keys]
        namespaced_keys_dict = {key: namespaced_key for key, namespaced_key in zip(
            keys, self._convert_list_of_keys_to_namespace_name(keys, io_type))}

        values_dict = {}
        
        for key, namespaced_key in namespaced_keys_dict.items():
            # new_key can be key or namespaced_key according to full_name value
            new_key = full_name * namespaced_key + (1 - full_name) * key
            if namespaced_key not in self.dm.data_id_map:
                raise Exception(
                    f'The key {namespaced_key} for the discipline {self.get_disc_full_name()} is missing in the data manager')
            # get data in local_data during run or linearize steps
            elif self.status in [self.STATUS_RUNNING, self.STATUS_LINEARIZE] and namespaced_key in self.mdo_discipline.local_data:
                values_dict[new_key] = self.mdo_discipline.local_data[namespaced_key]
            # get data in data manager during configure step
            else:
                values_dict[new_key] = self.dm.get_value(namespaced_key)

        return values_dict

    # -- execute/runtime section
#     def execute(self, input_data=None):
#         """
#         Overwrite execute method from MDODiscipline to load input_data from datamanager if possible
#         IMPORTANT NOTE: input_data should NOT be filled when execute called outside GEMS
#         """
#         if input_data is None:
#             # if no input_data, i.e. not called by GEMS
#             input_data = self.get_input_data_for_gems()
#         else:
#             # if input_data exists, i.e. called by GEMS
#             # no need to convert the data
#             pass
# 
#         # update status to 'PENDING' after configuration
#         self.update_status_pending()
# 
#         result = None
#         try:
#             result = MDODiscipline.execute(self, input_data=input_data)
#         except Exception as error:
#             # Update data manager status (status 'FAILED' is not propagate correctly due to exception
#             # so we have to force data manager status update in this case
#             self._update_status_dm(self.status)
#             raise error
# 
#         # When execution is done, is the status is again to 'pending' then we have to check if execution has been used
#         # If execution cache is used, then the discipline is not run and its
#         # status is not changed
#         if (self.status == ProxyDiscipline.STATUS_PENDING and self._cache_was_loaded is True):
#             self._update_status_recursive(self.STATUS_DONE)
# 
#         self.__check_nan_in_data(result)
# m
#         if self.check_min_max_couplings:
#             self.display_min_max_couplings()
#         return result

#     def linearize(self, input_data=None, force_all=False, force_no_exec=False,
#                   exec_before_linearize=True):
#         """overloads GEMS linearize function
#         """
#         # set GEM's default_inputs for gradient computation purposes
#         # to be deleted during GEMS update
# 
#         if input_data is not None:
#             self.default_inputs = input_data
#         else:
#             self.default_inputs = {}
#             input_data = self.get_input_data_for_gems()
#             self.default_inputs = input_data
# 
#         if self.linearization_mode == self.COMPLEX_STEP:
#             # is complex_step, switch type of inputs variables
#             # perturbed to complex
#             inputs, _ = self._retreive_diff_inouts(force_all)
#             def_inputs = self.default_inputs
#             for name in inputs:
#                 def_inputs[name] = def_inputs[name].astype('complex128')
#         else:
#             pass
# 
#         # need execution before the linearize
#         if not force_no_exec and exec_before_linearize:
#             self.reset_statuses_for_run()
#             self.exec_for_lin = True
#             self.execute(input_data)
#             self.exec_for_lin = False
#             force_no_exec = True
#             need_execution_after_lin = False
# 
#         # need execution but after linearize, in the NR GEMSEO case an
#         # execution is done bfore the while loop which udates the local_data of
#         # each discipline
#         elif not force_no_exec and not exec_before_linearize:
#             force_no_exec = True
#             need_execution_after_lin = True
# 
#         # no need of any execution
#         else:
#             need_execution_after_lin = False
#             # maybe no exec before the first linearize, GEMSEO needs a
#             # local_data with inputs and outputs for the jacobian computation
#             # if the local_data is empty
#             if self.local_data == {}:
#                 own_data = {
#                     k: v for k, v in input_data.items() if self.is_input_existing(k) or self.is_output_existing(k)}
#                 self.local_data = own_data
# 
#         if self.check_linearize_data_changes and not self.is_sos_coupling:
#             disc_data_before_linearize = {key: {'value': value} for key, value in deepcopy(
#                 input_data).items() if key in self.input_grammar.data_names}
# 
#         # set LINEARIZE status to get inputs from local_data instead of
#         # datamanager
#         self._update_status_dm(self.STATUS_LINEARIZE)
#         result = MDODiscipline.linearize(
#             self, input_data, force_all, force_no_exec)
#         # reset DONE status
#         self._update_status_dm(self.STATUS_DONE)
# 
#         self.__check_nan_in_data(result)
#         if self.check_linearize_data_changes and not self.is_sos_coupling:
#             disc_data_after_linearize = {key: {'value': value} for key, value in deepcopy(
#                 input_data).items() if key in disc_data_before_linearize.keys()}
#             is_output_error = True
#             output_error = self.check_discipline_data_integrity(disc_data_before_linearize,
#                                                                 disc_data_after_linearize,
#                                                                 'Discipline data integrity through linearize',
#                                                                 is_output_error=is_output_error)
#             if output_error != '':
#                 raise ValueError(output_error)
# 
#         if need_execution_after_lin:
#             self.reset_statuses_for_run()
#             self.execute(input_data)
# 
#         return result

#     def _get_columns_indices(self, inputs, outputs, input_column, output_column):
#         """
#         returns indices of input_columns and output_columns
#         """
#         # Get boundaries of the jacobian to compare
#         if inputs is None:
#             inputs = self.get_input_data_names()
#         if outputs is None:
#             outputs = self.get_output_data_names()
# 
#         indices = None
#         if input_column is not None or output_column is not None:
#             if len(inputs) == 1 and len(outputs) == 1:
# 
#                 if self.proxy_disciplines is not None:
#                     for discipline in self.proxy_disciplines:
#                         self.jac_boundaries.update(discipline.jac_boundaries)
# 
#                 indices = {}
#                 if output_column is not None:
#                     jac_bnd = self.jac_boundaries[f'{outputs[0]},{output_column}']
#                     tup = [jac_bnd['start'], jac_bnd['end']]
#                     indices[outputs[0]] = [i for i in range(*tup)]
# 
#                 if input_column is not None:
#                     jac_bnd = self.jac_boundaries[f'{inputs[0]},{input_column}']
#                     tup = [jac_bnd['start'], jac_bnd['end']]
#                     indices[inputs[0]] = [i for i in range(*tup)]
# 
#             else:
#                 raise Exception(
#                     'Not possible to use input_column and output_column options when \
#                     there is more than one input and output')
# 
#         return indices
# 
#     def set_partial_derivative(self, y_key, x_key, value):
#         '''
#         Set the derivative of y_key by x_key inside the jacobian of GEMS self.jac
#         '''
#         new_y_key = self.get_var_full_name(y_key, self._data_out)
# 
#         new_x_key = self.get_var_full_name(x_key, self._data_in)
# 
#         if new_x_key in self.jac[new_y_key]:
#             if isinstance(value, ndarray):
#                 value = lil_matrix(value)
#             self.jac[new_y_key][new_x_key] = value
# 
#     def set_partial_derivative_for_other_types(self, y_key_column, x_key_column, value):
#         '''
#         Set the derivative of the column y_key by the column x_key inside the jacobian of GEMS self.jac
#         y_key_column = 'y_key,column_name'
#         '''
#         if len(y_key_column) == 2:
#             y_key, y_column = y_key_column
#         else:
#             y_key = y_key_column[0]
#             y_column = None
# 
#         lines_nb_y, index_y_column = self.get_boundary_jac_for_columns(
#             y_key, y_column, self.IO_TYPE_OUT)
# 
#         if len(x_key_column) == 2:
#             x_key, x_column = x_key_column
#         else:
#             x_key = x_key_column[0]
#             x_column = None
# 
#         lines_nb_x, index_x_column = self.get_boundary_jac_for_columns(
#             x_key, x_column, self.IO_TYPE_IN)
# 
#         # Convert keys in namespaced keys in the jacobian matrix for GEMS
#         new_y_key = self.get_var_full_name(y_key, self._data_out)
# 
#         new_x_key = self.get_var_full_name(x_key, self._data_in)
# 
#         # Code when dataframes are filled line by line in GEMS, we keep the code for now
#         #         if index_y_column and index_x_column is not None:
#         #             for iy in range(value.shape[0]):
#         #                 for ix in range(value.shape[1]):
#         #                     self.jac[new_y_key][new_x_key][iy * column_nb_y + index_y_column,
#         # ix * column_nb_x + index_x_column] = value[iy, ix]
# 
#         if new_x_key in self.jac[new_y_key]:
#             if index_y_column is not None and index_x_column is not None:
#                 self.jac[new_y_key][new_x_key][index_y_column * lines_nb_y:(index_y_column + 1) * lines_nb_y,
#                                                index_x_column * lines_nb_x:(index_x_column + 1) * lines_nb_x] = value
#                 self.jac_boundaries.update({f'{new_y_key},{y_column}': {'start': index_y_column * lines_nb_y,
#                                                                         'end': (index_y_column + 1) * lines_nb_y},
#                                             f'{new_x_key},{x_column}': {'start': index_x_column * lines_nb_x,
#                                                                         'end': (index_x_column + 1) * lines_nb_x}})
# 
#             elif index_y_column is None and index_x_column is not None:
#                 self.jac[new_y_key][new_x_key][:, index_x_column * 
#                                                lines_nb_x:(index_x_column + 1) * lines_nb_x] = value
# 
#                 self.jac_boundaries.update({f'{new_y_key},{y_column}': {'start': 0,
#                                                                         'end':-1},
#                                             f'{new_x_key},{x_column}': {'start': index_x_column * lines_nb_x,
#                                                                         'end': (index_x_column + 1) * lines_nb_x}})
#             elif index_y_column is not None and index_x_column is None:
#                 self.jac[new_y_key][new_x_key][index_y_column * lines_nb_y:(index_y_column + 1) * lines_nb_y,
#                                                :] = value
#                 self.jac_boundaries.update({f'{new_y_key},{y_column}': {'start': index_y_column * lines_nb_y,
#                                                                         'end': (index_y_column + 1) * lines_nb_y},
#                                             f'{new_x_key},{x_column}': {'start': 0,
#                                                                         'end':-1}})
#             else:
#                 raise Exception(
#                     'The type of a variable is not yet taken into account in set_partial_derivative_for_other_types')
# 
#     def get_boundary_jac_for_columns(self, key, column, io_type):
#         data_io_disc = self.get_data_io_dict(io_type)
#         var_full_name = self.get_var_full_name(key, data_io_disc)
#         key_type = self.dm.get_data(var_full_name, self.TYPE)
#         value = self._get_sosdisc_io(key, io_type)[key]
#         
#         if key_type == 'dataframe':
#             # Get the number of lines and the index of column from the metadata
#             lines_nb = len(value)
#             index_column = [column for column in value.columns if column not in self.DEFAULT_EXCLUDED_COLUMNS].index(column)
#         elif key_type == 'array' or key_type == 'float':
#             lines_nb = None
#             index_column = None
#         elif key_type == 'dict':
#             dict_keys = list(value.keys())
#             lines_nb = len(value[column])
#             index_column = dict_keys.index(column)
# 
#         return lines_nb, index_column

    def get_input_data_for_gems(self):
        '''
        Get input_data for linearize ProxyDiscipline
        '''
        input_data = {}
        input_data_names = self.input_grammar.get_data_names()
        if len(input_data_names) > 0:

            for data_name in input_data_names:
                input_data[data_name] = self.ee.dm.get_value(data_name)

        return input_data

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

    def update_dm_with_local_data(self, local_data):
        '''
        Update the DM with local data from GEMSEO
        '''
        self.dm.set_values_from_dict(local_data)

    def run(self):
        ''' To be overloaded by sublcasses
        '''
        raise NotImplementedError()
    
    def set_proxy_status(self):
        '''
        Set proxy discipline status with mdo discipline status
        '''
        self._update_status_dm(self.mdo_discipline.status)
        for proxy_disc in self.proxy_disciplines:
            proxy_disc.set_proxy_status()

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

    def store_sos_outputs_values(self, dict_values, update_dm=False):
        ''' store outputs from 'dict_values' into self._data_out
        '''
        # fill data using data connector if needed
        self._update_with_values(self._data_out, dict_values, update_dm)

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
        
    def _set_dm_cache_map(self):
        '''
        Update cache_map dict in DM with cache and its children recursively
        '''
        if self.cache is not None:
            self._store_cache_with_hashed_uid(self)
        # store children cache recursively
        for disc in self.proxy_disciplines:
            disc._set_dm_cache_map() 
            
    def _store_cache_with_hashed_uid(self, disc):
        '''
        Generate hashed uid and store cache in DM
        '''
        full_name = self.get_disc_full_name().split(self.ee.study_name)[-1]
        class_name = disc.__class__.__name__
        anoninmated_data_io = self.get_anonimated_data_io(disc)
        
        # set disc infos string list with full name, class name and anonimated i/o for hashed uid generation
        disc_info_list = [full_name, class_name, anoninmated_data_io]
        hashed_uid = self.dm.generate_hashed_uid(disc_info_list)
        
        # store cache in DM map
        self.dm.cache_map[hashed_uid] = disc.cache
        
        # store disc in DM map
        if hashed_uid in self.dm.gemseo_disciplines_id_map:
            self.dm.gemseo_disciplines_id_map[hashed_uid].append(disc)
        else:
            self.dm.gemseo_disciplines_id_map[hashed_uid] = [disc]
            
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

            try:
                var_f_name = self.get_var_full_name(var_name, self._data_in)
                default_val = self.dm.data_dict[self.dm.get_data_id(
                    var_f_name)][self.DEFAULT]
            except:
                var_f_name = self.get_var_full_name(var_name, self._data_in)
            if self.dm.get_value(var_f_name) is None and default_val is not None:
                self._data_in[var_name][self.VALUE] = default_val
            else:
                # update from dm for all proxy_disciplines to load all data
                self._data_in[var_name][self.VALUE] = self.dm.get_value(
                    var_f_name)
        # -- update sub-disciplines
        for discipline in self.proxy_disciplines:
            discipline.update_from_dm()

    # -- Ids and namespace handling
    def get_disc_full_name(self):
        '''
        Returns the discipline name with full namespace
        '''
        return self.ee.ns_manager.get_local_namespace_value(self)

    def get_disc_id_from_namespace(self):

        return self.ee.dm.get_discipline_ids_list(self.get_disc_full_name())
    
    def get_anonimated_data_io(self, disc):
        '''
        return list of anonimated input and output keys for serialisation purpose
        '''
        anonimated_data_io = ''

        for key in disc.get_input_data_names():
            anonimated_data_io += key.split(self.ee.study_name)[-1]
            
        for key in disc.get_output_data_names():
            anonimated_data_io += key.split(self.ee.study_name)[-1]

        return anonimated_data_io

    def _convert_list_of_keys_to_namespace_name(self, keys, io_type):

        # Refactor  variables keys with namespace
        if isinstance(keys, list):
            variables = [self._convert_to_namespace_name(
                key, io_type) for key in keys]
        else:
            variables = [self._convert_to_namespace_name(
                keys, io_type)]
        return variables

#     def _init_grammar_with_keys(self, names, io_type):
#         ''' initialize GEMS grammar with names and type None
#         '''
#         names_dict = dict.fromkeys(names, None)
#         if io_type == self.IO_TYPE_IN:
#             grammar = self.input_grammar
#             grammar.clear()
# 
#         elif io_type == self.IO_TYPE_OUT:
#             grammar = self.output_grammar
#             grammar.clear()
#         grammar.initialize_from_base_dict(names_dict)
# 
#         return grammar

    def _convert_to_namespace_name(self, key, io_type):
        ''' Convert to namepsace with coupling_namespace management
            Using a key (variables name) and reference_data (yaml in or out),
            build the corresponding namespaced key using the visibility property
        '''

        # Refactor  variables keys with namespace
        result = self.ee.ns_manager.get_namespaced_variable(
            self, key, io_type)

        return result

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
        for disc in self.proxy_disciplines:
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

    # -- Maturity handling section
    def set_maturity(self, maturity, maturity_dict=False):
        if maturity is None or maturity in self.possible_maturities or maturity_dict:
            self._maturity = maturity
        else:
            raise Exception(
                f'Unkown maturity {maturity} for discipline {self.sos_name}')

    def get_maturity(self):
        '''
        Get the maturity of the ProxyDiscipline, a discipline does not have any subdisciplines, only a coupling has
        '''
        if hasattr(self, '_maturity'):
            return self._maturity
        else:
            return ''

    def _build_dynamic_DESC_IN(self):
        pass

    def _convert_new_type_into_array(
            self, var_dict, update_dm=True):
        '''
        Check element type in var_dict, convert new type into numpy array
            and stores metadata into DM for afterwards reconversion
        '''
        # dm_reduced = self.dm.convert_data_dict_with_full_name()
        # dm_reduced = self.dm.get_data_dict_list_attr([self.VAR_TYPE_ID, self.DF_EXCLUDED_COLUMNS, self.TYPE_METADATA])
        var_dict_converted, dict_to_update_dm = convert_new_type_into_array(
            var_dict, self.dm)

        # update dm
        if update_dm:
            for key in dict_to_update_dm.keys():
                self.dm.set_data(key, self.TYPE_METADATA,
                                 dict_to_update_dm[key], check_value=False)

        return var_dict_converted

    def _convert_array_into_new_type(self, local_data):
        """ convert list in local_data into correct type in data_in
            returns an updated copy of local_data
        """

        # dm_reduced = self.dm.get_data_dict_list_attr([self.VAR_TYPE_ID, self.DF_EXCLUDED_COLUMNS, self.TYPE_METADATA])
        return convert_array_into_new_type(local_data, self.dm)

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
            return not dict_are_equal(dict_values_dm,
                         self._structuring_variables)

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
            has_nan = self.__check_nan_in_data_rec(data, "")
            if has_nan:
                raise ValueError(f'NaN values found in {self.sos_name}')

    def __check_nan_in_data_rec(self, data, parent_key):
        """ Using entry data, check if nan value exist in data's as recursive
        method

        :params: data
        :type: composite data

        :params: parent_key, on composite type (dict), reference parent key
        :type: str

        """
        has_nan = False
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
                has_nan = True
        return has_nan

#     def check_jacobian(self, input_data=None, derr_approx=MDODiscipline.FINITE_DIFFERENCES,
#                        step=1e-7, threshold=1e-8, linearization_mode='auto',
#                        inputs=None, outputs=None, parallel=False,
#                        n_processes=MDODiscipline.N_CPUS,
#                        use_threading=False, wait_time_between_fork=0,
#                        auto_set_step=False, plot_result=False,
#                        file_path="jacobian_errors.pdf",
#                        show=False, figsize_x=10, figsize_y=10, input_column=None, output_column=None,
#                        dump_jac_path=None, load_jac_path=None):
#         """
#         Overload check jacobian to execute the init_execution
#         """
# 
#         # The init execution allows to check jacobian without an execute before the check
#         # however if an execute was done, we do not want to restart the model
#         # and potentially loose informations to compute gradients (some
#         # gradients are computed with the model)
#         if self.status != self.STATUS_DONE:
#             self.init_execution()
# 
#         # if dump_jac_path is provided, we trigger GEMSEO dump
#         if dump_jac_path is not None:
#             reference_jacobian_path = dump_jac_path
#             save_reference_jacobian = True
#         # if dump_jac_path is provided, we trigger GEMSEO dump
#         elif load_jac_path is not None:
#             reference_jacobian_path = load_jac_path
#             save_reference_jacobian = False
#         else:
#             reference_jacobian_path = None
#             save_reference_jacobian = False
# 
#         approx = DisciplineJacApprox(
#             self,
#             derr_approx,
#             step,
#             parallel,
#             n_processes,
#             use_threading,
#             wait_time_between_fork,
#         )
#         if inputs is None:
#             inputs = self.get_input_data_names(filtered_inputs=True)
#         if outputs is None:
#             outputs = self.get_output_data_names(filtered_outputs=True)
# 
#         if auto_set_step:
#             approx.auto_set_step(outputs, inputs, print_errors=True)
# 
#         # Differentiate analytically
#         self.add_differentiated_inputs(inputs)
#         self.add_differentiated_outputs(outputs)
#         self.linearization_mode = linearization_mode
#         self.reset_statuses_for_run()
#         # Linearize performs execute() if needed
#         self.linearize(input_data)
# 
#         if input_column is None and output_column is None:
#             indices = None
#         else:
#             indices = self._get_columns_indices(
#                 inputs, outputs, input_column, output_column)
# 
#         jac_arrays = {
#             key_out: {key_in: value.toarray() if not isinstance(value, ndarray) else value for key_in, value in
#                       subdict.items()}
#             for key_out, subdict in self.jac.items()}
#         o_k = approx.check_jacobian(
#             jac_arrays,
#             outputs,
#             inputs,
#             self,
#             threshold,
#             plot_result=plot_result,
#             file_path=file_path,
#             show=show,
#             figsize_x=figsize_x,
#             figsize_y=figsize_y,
#             reference_jacobian_path=reference_jacobian_path,
#             save_reference_jacobian=save_reference_jacobian,
#             indices=indices,
#         )
#         return o_k

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

    def display_min_max_couplings(self):
        ''' Method to display the minimum and maximum values among a discipline's couplings

        '''
        min_coupling_dict, max_coupling_dict = {}, {}
        for key, value in self.mdo_discipline.local_data.items():
            is_coupling = self.dm.get_data(key, 'coupling')
            if is_coupling:
                min_coupling_dict[key] = min(abs(value))
                max_coupling_dict[key] = max(abs(value))
        min_coupling = min(min_coupling_dict, key=min_coupling_dict.get)
        max_coupling = max(max_coupling_dict, key=max_coupling_dict.get)
        self.ee.logger.info(
            "in discipline <%s> : <%s> has the minimum coupling value <%s>" % (
                self.sos_name, min_coupling, min_coupling_dict[min_coupling]))
        self.ee.logger.info(
            "in discipline <%s> : <%s> has the maximum coupling value <%s>" % (
                self.sos_name, max_coupling, max_coupling_dict[max_coupling]))

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
