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
import copy

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

from gemseo.utils.compare_data_manager_tooling import dict_are_equal
from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.data_connector.data_connector_factory import ConnectorFactory

from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import convert_array_into_new_type, \
    convert_new_type_into_array

from gemseo.core.discipline import MDODiscipline
from sostrades_core.execution_engine.mdo_discipline_wrapp import MDODisciplineWrapp
from sostrades_core.execution_engine.sos_mdo_discipline import SoSMDODiscipline
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from gemseo.core.chain import MDOChain
from sostrades_core.tools.controllers.simpy_formula import SympyFormula
from sostrades_core.tools.check_data_integrity.check_data_integrity import CheckDataIntegrity


class ProxyDisciplineException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'


class ProxyDiscipline(object):
    """
    **ProxyDiscipline** is a class proxy for a  discipline on the SoSTrades side.

    It contains the information and methonds necessary for i/o configuration (static or dynamic).

    Leaves of the process tree are direct instances of ProxyDiscipline. Other nodes are instances that inherit from
    ProxyDiscipline (e.g. ProxyCoupling).

    An instance of ProxyDiscipline is in one-to-one aggregation with an instance of MDODisciplineWrapp, which allows the
    use of different wrapping modes to provide the model run.

    During the prepare_execution step, the ProxyDiscipline coordinates the instantiation of the GEMSEO objects that
    manage the model run.

    Attributes:
        mdo_discipline_wrapp (MDODisciplineWrapp): aggregated object that references the wrapper and GEMSEO discipline

        proxy_disciplines (List[ProxyDiscipline]): children in the process tree
        status (property,<<associated with string _status>>): status in the current process,either CONFIGURATION or
         provided by the GEMSEO objects during run

        disc_id (string): anonymized discipline identifier in the data manager
        sos_name (string): name of the discipline/node
        ee (ExecutionEngine): execution engine of the process
        dm (DataManager): data manager of the process


        is_sos_coupling (bool): type of node flag
        is_optim_scenario (bool): type of node flag
        is_parallel (bool): type of node flag
        is_specific_driver (bool): type of node flag

        _is_configured (bool): flag for configuration relaying on children configuration and structuring vars changes
        _reset_cache (bool): flag to reset cache

        inst_desc_in (Dict[Dict]): desc_in of instance used to add dynamic inputs
        inst_desc_out (Dict[Dict]): desc_out of instance used to add dynamic outputs
        _data_in (Dict[Dict]): instance variable for input data handling containing description of variables in disc and subprocess
        _data_out (Dict[Dict]): instance variable for output data handling containing description of variables in disc and subprocess

        _io_ns_map_in(Dict[int]): map of short names to namespace object id of discipline DESC_IN+NUM_DESC_IN+inst_desc_in
        _io_ns_map_out(Dict[int]): map of short names to namespace object id of discipline DESC_OUT+inst_desc_out

        _structuring_variables (Dict[Any]): stored values of variables whose changes force revert of the configured status
        _maturity (string): maturity of the user-defined model


        cls (Class): constructor of the model wrapper with user-defin ed run (or None)
    """
    # -- Disciplinary attributes
    DESC_IN = None
    DESC_OUT = None
    IO_TYPE = 'io_type'
    IO_TYPE_IN = SoSWrapp.IO_TYPE_IN
    IO_TYPE_OUT = SoSWrapp.IO_TYPE_OUT
    TYPE = SoSWrapp.TYPE
    SUBTYPE = SoSWrapp.SUBTYPE
    COUPLING = SoSWrapp.COUPLING
    VISIBILITY = SoSWrapp.VISIBILITY
    LOCAL_VISIBILITY = SoSWrapp.LOCAL_VISIBILITY
    INTERNAL_VISIBILITY = SoSWrapp.INTERNAL_VISIBILITY
    SHARED_VISIBILITY = SoSWrapp.SHARED_VISIBILITY
    AVAILABLE_VISIBILITIES = [
        LOCAL_VISIBILITY,
        INTERNAL_VISIBILITY,
        SHARED_VISIBILITY]
    NAMESPACE = SoSWrapp.NAMESPACE
    NS_REFERENCE = 'ns_reference'
    VALUE = SoSWrapp.VALUE
    DEFAULT = SoSWrapp.DEFAULT
    EDITABLE = SoSWrapp.EDITABLE
    USER_LEVEL = SoSWrapp.USER_LEVEL
    STRUCTURING = SoSWrapp.STRUCTURING
    POSSIBLE_VALUES = SoSWrapp.POSSIBLE_VALUES
    RANGE = SoSWrapp.RANGE
    UNIT = SoSWrapp.UNIT
    DESCRIPTION = SoSWrapp.DESCRIPTION
    NUMERICAL = SoSWrapp.NUMERICAL
    META_INPUT = 'meta_input'
    OPTIONAL = 'optional'
    ORIGIN = 'model_origin'
    HEADERS = 'headers'
    COMPOSED_OF = 'composed_of'
    DISCIPLINES_DEPENDENCIES = 'disciplines_dependencies'
    VAR_NAME = SoSWrapp.VAR_NAME
    VISIBLE = SoSWrapp.VISIBLE
    CONNECTOR_DATA = SoSWrapp.CONNECTOR_DATA
    CACHE_TYPE = 'cache_type'
    CACHE_FILE_PATH = 'cache_file_path'
    FORMULA = 'formula'
    IS_FORMULA = 'is_formula'
    IS_EVAL = 'is_eval'
    CHECK_INTEGRITY_MSG = 'check_integrity_msg'

    DATA_TO_CHECK = [TYPE, UNIT, RANGE,
                     POSSIBLE_VALUES, USER_LEVEL]
    NO_UNIT_TYPES = ['bool', 'string', 'string_list']
    # Dict  ex: {'ColumnName': (column_data_type, column_data_range,
    # column_editable)}
    DATAFRAME_DESCRIPTOR = SoSWrapp.DATAFRAME_DESCRIPTOR
    DATAFRAME_EDITION_LOCKED = SoSWrapp.DATAFRAME_EDITION_LOCKED
    #
    DF_EXCLUDED_COLUMNS = 'dataframe_excluded_columns'
    DEFAULT_EXCLUDED_COLUMNS = ['year', 'years']
    DISCIPLINES_FULL_PATH_LIST = 'discipline_full_path_list'

    # -- Variable types information section
    VAR_TYPE_ID = 'type'
    # complex can also be a type if we use complex step
    INT_MAP = (int, np_int32, np_int64, np_complex128)
    FLOAT_MAP = (float, np_float64, np_complex128)
    PROC_BUILDER_MODAL = 'proc_builder_modal'
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
    NEW_VAR_TYPE = ['dict', 'dataframe',
                    'string_list', 'string', 'float', 'int', 'list']

    UNSUPPORTED_GEMSEO_TYPES = []
    for type in VAR_TYPE_MAP.keys():
        if type not in VAR_TYPE_GEMS and type not in NEW_VAR_TYPE:
            UNSUPPORTED_GEMSEO_TYPES.append(type)

    # # Warning : We cannot put string_list into dict, all other types inside a dict are possiblr with the type dict
    # # df_dict = dict , string_dict = dict, list_dict = dict
    TYPE_METADATA = "type_metadata"

    DEFAULT = 'default'
    POS_IN_MODE = ['value', 'list', 'dict']

    DEBUG_MODE = SoSMDODiscipline.DEBUG_MODE
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
                     POSSIBLE_VALUES: ['None', MDODiscipline.SIMPLE_CACHE],
                     # ['None', MDODiscipline.SIMPLE_CACHE, MDODiscipline.HDF5_CACHE, MDODiscipline.MEMORY_FULL_CACHE]
                     NUMERICAL: True,
                     STRUCTURING: True},
        CACHE_FILE_PATH: {TYPE: 'string', DEFAULT: '', NUMERICAL: True, OPTIONAL: True, STRUCTURING: True},
        DEBUG_MODE: {TYPE: 'string', DEFAULT: '', POSSIBLE_VALUES: list(AVAILABLE_DEBUG_MODE),
                     NUMERICAL: True, STRUCTURING: True}
    }

    # -- grammars
    SOS_GRAMMAR_TYPE = "SoSSimpleGrammar"

    # -- status
    STATUS_VIRTUAL = MDODiscipline.STATUS_VIRTUAL
    STATUS_PENDING = MDODiscipline.STATUS_PENDING
    STATUS_DONE = MDODiscipline.STATUS_DONE
    STATUS_RUNNING = MDODiscipline.STATUS_RUNNING
    STATUS_FAILED = MDODiscipline.STATUS_FAILED
    STATUS_CONFIGURE = MDODiscipline.STATUS_CONFIGURE
    STATUS_LINEARIZE = MDODiscipline.STATUS_LINEARIZE

    EE_PATH = 'sostrades_core.execution_engine'

    def __init__(self, sos_name, ee, cls_builder=None, associated_namespaces=None):
        '''
        Constructor

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
            cls_builder (Class): class constructor of the user-defined wrapper (or None)
            associated_namespaces(List[string]): list containing ns ids ['name__value'] for namespaces associated to builder
        '''
        # Enable not a number check in execution result and jacobian result
        # Be carreful that impact greatly calculation performances
        self.mdo_discipline_wrapp = None
        self.create_mdo_discipline_wrap(
            name=sos_name, wrapper=cls_builder, wrapping_mode='SoSTrades')
        self._reload(sos_name, ee, associated_namespaces)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.Discipline')
        self.model = None
        self.father_builder = None
        self.father_executor = None
        self.cls = cls_builder

    def set_father_executor(self, father_executor):
        """
        set father executor

        Arguments:
            father_executor (ProxyDiscipline): proxy that orchestrates the execution of this proxy discipline (e.g. coupling)
        """
        self.father_executor = father_executor

    def _reload(self, sos_name, ee, associated_namespaces=None):
        """
        Reload ProxyDiscipline attributes and set is_sos_coupling.

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
            associated_namespaces(List[string]): list containing ns ids ['name__value'] for namespaces associated to builder
        """
        self.proxy_disciplines = []
        self._status = None
        self.status_observers = []
        self.__config_dependency_disciplines = []
        self.__config_dependent_disciplines = []
        # -- Base disciplinary attributes
        self.jac_boundaries = {}
        self.disc_id = None
        self.sos_name = sos_name
        self.ee = ee
        self.dm = self.ee.dm
        if associated_namespaces is None:
            self.associated_namespaces = []
        else:
            self.associated_namespaces = associated_namespaces
        self.ee.ns_manager.create_disc_ns_info(self)

        if not hasattr(self, 'is_sos_coupling'):
            self.is_sos_coupling = False
        self.is_optim_scenario = False
        self.is_parallel = False
        self.is_specific_driver = False

        self.in_checkjac = False
        self._is_configured = False
        self._reset_cache = False
        self._set_children_cache = False
        self._reset_debug_mode = False

        # -- disciplinary data attributes
        self.inst_desc_in = None  # desc_in of instance used to add dynamic inputs
        self.inst_desc_out = None  # desc_out of instance used to add dynamic outputs

        self._data_in = None
        self._data_out = None
        self._io_ns_map_in = None
        self._io_ns_map_out = None  # used by ProxyCoupling, ProxyDriverEvaluator

        self._structuring_variables = None
        self.reset_data()
        # -- Maturity attribute
        self._maturity = self.get_maturity()

        # Add the discipline in the dm and get its unique disc_id (was in the
        # configure)
        self._set_dm_disc_info()

        # Instantiate check_data_integrity class to check data after dm save
        self.check_data_integrity_cls = CheckDataIntegrity(
            self.__class__, self.dm)
        # update discipline status to CONFIGURE
        self._update_status_dm(self.STATUS_CONFIGURE)

    def create_mdo_discipline_wrap(self, name, wrapper, wrapping_mode):
        """
        creation of mdo_discipline_wrapp by the proxy
        To be overloaded by proxy without MDODisciplineWrapp (eg scatter...)
        """
        self.mdo_discipline_wrapp = MDODisciplineWrapp(
            name, wrapper, wrapping_mode)

    @property
    def status(self):  # type: (...) -> str
        """
        The status of the discipline, to be retrieved from the GEMSEO object after configuration.
        """
        if self._status != self.STATUS_CONFIGURE:
            return self.get_status_after_configure()
        return self.STATUS_CONFIGURE

    @property
    def config_dependency_disciplines(self):  # type: (...) -> str
        """
        The config_dependency_disciplines list which represents the list of disciplines that must be configured before you configure
        """

        return self.__config_dependency_disciplines

    @property
    def config_dependent_disciplines(self):  # type: (...) -> str
        """
        The config_dependent_disciplines list which represents the list of disciplines that need to be configured after you configure
        """

        return self.__config_dependent_disciplines

    @status.setter
    def status(self, status):
        """
        setter of status
        """
        self._update_status_dm(status)

    def prepare_execution(self):
        '''
        GEMSEO objects instanciation
        '''
        if self.mdo_discipline_wrapp.mdo_discipline is None:
            # init gemseo discipline if it has not been created yet
            self.mdo_discipline_wrapp.create_gemseo_discipline(proxy=self,
                                                               reduced_dm=self.ee.dm.reduced_dm,
                                                               cache_type=self.get_sosdisc_inputs(
                                                                   self.CACHE_TYPE),
                                                               cache_file_path=self.get_sosdisc_inputs(
                                                                   self.CACHE_FILE_PATH))
            self.add_status_observers_to_gemseo_disc()

        else:
            # TODO : this should only be necessary when changes in structuring
            # variables happened?
            self.set_wrapper_attributes(self.mdo_discipline_wrapp.wrapper)

            if self._reset_cache:
                # set new cache when cache_type have changed (self._reset_cache
                # == True)
                self.set_cache(self.mdo_discipline_wrapp.mdo_discipline, self.get_sosdisc_inputs(self.CACHE_TYPE),
                               self.get_sosdisc_inputs(self.CACHE_FILE_PATH))
                if self.get_sosdisc_inputs(self.CACHE_TYPE) == 'None' and self.dm.cache_map is not None:
                    self.delete_cache_in_cache_map()

#             if self._reset_debug_mode:
#                 # update default values when changing debug modes between executions
#                 to_update_debug_mode = self.get_sosdisc_inputs(self.DEBUG_MODE, in_dict=True, full_name=True)
#                 self.mdo_discipline_wrapp.update_default_from_dict(to_update_debug_mode)
            # set the status to pending on GEMSEO side (so that it does not
            # stay on DONE from last execution)
            self.mdo_discipline_wrapp.mdo_discipline.status = MDODiscipline.STATUS_PENDING
        self.status = self.mdo_discipline_wrapp.mdo_discipline.status
        self._reset_cache = False
        self._reset_debug_mode = False

    def add_status_observers_to_gemseo_disc(self):
        '''
        Add all observers that have been addes when gemseo discipline was not instanciated
        '''

        for observer in self.status_observers:
            if self.mdo_discipline_wrapp is not None and self.mdo_discipline_wrapp.mdo_discipline is not None:
                self.mdo_discipline_wrapp.mdo_discipline.add_status_observer(
                    observer)

    def set_cache(self, disc, cache_type, cache_hdf_file):
        '''
        Instanciate and set cache for disc if cache_type is not 'None'

        Arguments:
            disc (MDODiscipline): GEMSEO object to set cache
            cache_type (string): type of cache
            cache_hdf_file (string): cache hdf file path
        '''
        if cache_type == MDOChain.HDF5_CACHE and cache_hdf_file is None:
            raise Exception(
                'if the cache type is set to HDF5Cache, the cache_file path must be set')
        else:
            disc.cache = None
            if cache_type != 'None':
                disc.set_cache_policy(
                    cache_type=cache_type, cache_hdf_file=cache_hdf_file)

    def delete_cache_in_cache_map(self):
        '''
        If a cache has been written
        '''
        hashed_uid = self.get_cache_map_hashed_uid(self)
        self.dm.delete_hashed_id_in_cache_map(hashed_uid)

    def get_shared_namespace_list(self, data_dict):
        '''
        Get the list of namespaces defined in the data_in or data_out when the visibility of the variable is shared

        Arguments:
            data_dict (Dict[dict]): data_in or data_out
        '''
        shared_namespace_list = []

        for item in data_dict.values():
            self.__append_item_namespace(item, shared_namespace_list)

        return list(set(shared_namespace_list))

    def __append_item_namespace(self, item, ns_list):
        '''
        Append the namespace if the visibility is shared

        Arguments:
            item (dict): element to append to the ns_list
            ns_list (List[Namespace]): list of namespaces [???]
        '''
        if self.VISIBILITY in item and item[self.VISIBILITY] == self.SHARED_VISIBILITY:
            ns_list.append(item[self.NAMESPACE])

    def get_input_data_names(self, as_namespaced_tuple=False):
        '''
        Returns:
            (List[string]) of input data full names based on i/o and namespaces declarations in the user wrapper
        '''
        return list(self.get_data_io_with_full_name(self.IO_TYPE_IN, as_namespaced_tuple).keys())

    def get_output_data_names(self, as_namespaced_tuple=False):
        '''
        Returns:
            (List[string]) outpput data full names based on i/o and namespaces declarations in the user wrapper
        '''
        return list(self.get_data_io_with_full_name(self.IO_TYPE_OUT, as_namespaced_tuple).keys())

    def get_data_io_dict(self, io_type):
        '''
        Get the DESC_IN+NUM_DESC_IN+inst_desc_in or the DESC_OUT+inst_desc_out depending on the io_type

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT

        Returns:
            (Dict(dict)) data_in or data_out
        Raises:
            Exception if io_type
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
        Get the DESC_IN+NUM_DESC_IN+inst_desc_in or the DESC_OUT+inst_desc_out keys depending on the io_type

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT

        Returns:
            (dict_keys) data_in or data_out keys
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
        Return the namespace and the data_in/data_out of a single variable (short name)

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
            var_name (string): short name of the variable
        Returns:
            (dict) data_in or data_out of the variable
        '''
        data_dict_list = [d for k, d in self.get_data_io_dict(
            io_type).items() if k == var_name]
        if len(data_dict_list) != 1:
            raise Exception(
                f'No key matching with variable name {var_name} in the data_{io_type}')

        return data_dict_list[0]

    def get_variable_name_from_ns_key(self, io_type, ns_key):
        """
        """
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
        """
        Update data manager for this discipline with data_dict.

        Arguments:
            data_dict (Dict[dict]): item to update data manager with
        """
        self.dm.update_with_discipline_dict(
            self.disc_id, data_dict)

    def create_data_io_from_desc_io(self):
        """
        Create data_in and data_out from DESC_IN and DESC_OUT if empty
        """
        if self._data_in == {}:
            desc_in = self.get_desc_in_out(self.IO_TYPE_IN)
            self.set_shared_namespaces_dependencies(desc_in)
            desc_in = self._prepare_data_dict(self.IO_TYPE_IN, desc_in)
            # TODO: check if it is OK to update dm during config. rather than
            # at the very end of it (dynamic ns)
            self.update_dm_with_data_dict(desc_in)
            inputs_var_ns_tuples = self._extract_var_ns_tuples(desc_in)
            self._update_io_ns_map(inputs_var_ns_tuples, self.IO_TYPE_IN)
            self._update_data_io(
                zip(inputs_var_ns_tuples, desc_in.values()), self.IO_TYPE_IN)
            # Deal with numerical parameters inside the sosdiscipline
            self.add_numerical_param_to_data_in()

        if self._data_out == {}:
            desc_out = self.get_desc_in_out(self.IO_TYPE_OUT)
            self.set_shared_namespaces_dependencies(desc_out)
            desc_out = self._prepare_data_dict(self.IO_TYPE_OUT, desc_out)
            self.update_dm_with_data_dict(desc_out)
            outputs_var_ns_tuples = self._extract_var_ns_tuples(desc_out)
            self._update_io_ns_map(outputs_var_ns_tuples, self.IO_TYPE_OUT)
            self._update_data_io(
                zip(outputs_var_ns_tuples, desc_out.values()), self.IO_TYPE_OUT)

    def get_desc_in_out(self, io_type):
        """
        Retrieves information from wrapper or ProxyDiscipline DESC_IN to fill data_in
        To be overloaded by special proxies ( coupling, scatter,...)

        Argument:
            io_type : 'string' . indicates whether we are interested in desc_in or desc_out
        """
        if io_type == self.IO_TYPE_IN:
            return deepcopy(self.mdo_discipline_wrapp.wrapper.DESC_IN) or {}
        elif io_type == self.IO_TYPE_OUT:
            return deepcopy(self.mdo_discipline_wrapp.wrapper.DESC_OUT) or {}
        else:
            raise Exception(
                f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]')

    def _extract_var_ns_tuples(self, short_name_data_dict):
        """
        Exctracts tuples in the form (var_name, id(ns_ref)) for the variables in a complete data dictionary with variable
        short name as keys and variable description dictionary as values (i.e. like the DESC_IN etc. after calling
        _prepare_data_dict).

        Arguments:
            short_name_data_dict (Dict[Dict]): data dictionary as described above

        Returns:
            list[tuple] : [(var_short_name, id(ns_ref)), ...]
        """
        try:
            return list(zip(short_name_data_dict.keys(), [id(v[self.NS_REFERENCE]) for v in short_name_data_dict.values()]))
        except:
            print('vuhdidi')

    def _update_io_ns_map(self, var_ns_tuples, io_type):
        """
        Updates the variable _io_ns_map_in/_io_ns_map_out in the form {'var_short_name': id(ns_ref)}.

        Arguments:
            var_ns_tuples (list[tuple]): the tuples (var_short_name, id(ns_ref)) for the variables to add
            io_type (string) : IO_TYPE_IN or IO_TYPE_OUT

        Raises:
            Exception if io_type is not IO_TYPE_IN or IO_TYPE_OUT
        """
        if io_type == self.IO_TYPE_IN:
            self._io_ns_map_in.update(var_ns_tuples)
        elif io_type == self.IO_TYPE_OUT:
            self._io_ns_map_out.update(var_ns_tuples)
        else:
            raise Exception(
                f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]')

    def _restart_data_io_to_disc_io(self, io_type=None):
        """
        Restarts the _data_in/_data_out to contain only the variables referenced in short names in _io_ns_map_in/_io_ns_map_out.

        Arguments:
            io_type (string) : IO_TYPE_IN or IO_TYPE_OUT

        Raises:
            Exception if io_type is not IO_TYPE_IN or IO_TYPE_OUT
        """
        io_types = []
        if io_type is None:
            io_types = [self.IO_TYPE_IN, self.IO_TYPE_OUT]
        elif io_type == self.IO_TYPE_IN or io_type == self.IO_TYPE_OUT:
            io_types = [io_type]
        else:
            raise Exception(
                f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]')
        if self.IO_TYPE_IN in io_types:
            self._data_in = {(key, id(
                value[self.NS_REFERENCE])): value for key, value in self.get_data_in().items()}
        if self.IO_TYPE_OUT in io_types:
            self._data_out = {(key, id(
                value[self.NS_REFERENCE])): value for key, value in self.get_data_out().items()}

    def _update_data_io(self, data_dict, io_type, data_dict_in_short_names=False):
        """
        Updates the _data_in/_data_out with the variables described in the data_dict.

        Arguments:
            data_dict (dict[dict]): description of the variables to update with
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
            data_dict_in_short_names (bool): whether the keys of the data_dict are strings var_short_name (True) or tuples
                                            (var_short_name, id(ns_ref)) as in the _data_in/_data_out.
        Raises:
            Exception if io_type is not IO_TYPE_IN or IO_TYPE_OUT
        """
        if io_type == self.IO_TYPE_IN:
            data_io = self._data_in
        elif io_type == self.IO_TYPE_OUT:
            data_io = self._data_out
        else:
            raise Exception(
                f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]')

        if data_dict_in_short_names:
            error_msg = 'data_dict_in_short_names for uodate_data_io not implemented'
            self.logger.error(error_msg)
            raise Exception(error_msg)
#             data_io.update(zip(self._extract_var_ns_tuples(data_dict, io_type),  # keys are ns tuples
# data_dict.values()))                             # values are values
# i.e. var dicts
        else:
            data_io.update(data_dict)

    def add_numerical_param_to_data_in(self):
        """
        Add numerical parameters to the data_in
        """
        num_data_in = deepcopy(self.NUM_DESC_IN)
        num_data_in = self._prepare_data_dict(
            self.IO_TYPE_IN, data_dict=num_data_in)
        num_inputs_var_ns_tuples = self._extract_var_ns_tuples(num_data_in)
        self._update_io_ns_map(num_inputs_var_ns_tuples, self.IO_TYPE_IN)
        self.update_dm_with_data_dict(num_data_in)
        self._update_data_io(zip(num_inputs_var_ns_tuples,
                                 num_data_in.values()), self.IO_TYPE_IN)

    def update_data_io_with_inst_desc_io(self):
        """
        Update data_in and data_out with inst_desc_in and inst_desc_out
        """
        new_inputs = {}
        new_outputs = {}
        for key, value in self.inst_desc_in.items():
            if not key in self.get_data_in().keys():
                new_inputs[key] = value
        for key, value in self.inst_desc_out.items():
            if not key in self.get_data_out().keys():
                new_outputs[key] = value
        if len(new_inputs) > 0:
            self.set_shared_namespaces_dependencies(new_inputs)
            completed_new_inputs = self._prepare_data_dict(
                self.IO_TYPE_IN, new_inputs)
            self.update_dm_with_data_dict(
                completed_new_inputs)
            inputs_var_ns_tuples = self._extract_var_ns_tuples(
                completed_new_inputs)
            self._update_io_ns_map(inputs_var_ns_tuples, self.IO_TYPE_IN)
            self._update_data_io(
                zip(inputs_var_ns_tuples, completed_new_inputs.values()), self.IO_TYPE_IN)

        # add new outputs from inst_desc_out to data_out
        if len(new_outputs) > 0:
            self.set_shared_namespaces_dependencies(new_outputs)
            completed_new_outputs = self._prepare_data_dict(
                self.IO_TYPE_OUT, new_outputs)
            self.update_dm_with_data_dict(
                completed_new_outputs)
            outputs_var_ns_tuples = self._extract_var_ns_tuples(
                completed_new_outputs)
            self._update_io_ns_map(outputs_var_ns_tuples, self.IO_TYPE_OUT)
            self._update_data_io(
                zip(outputs_var_ns_tuples, completed_new_outputs.values()), self.IO_TYPE_OUT)

    def get_built_disciplines_ids(self):
        """
        Returns: (List[string]) the names of the sub proxies.
        """
        return [disc.name for disc in self.proxy_disciplines]

    def get_proxy_disciplines(self):
        """
        Returns: (List[ProxyDiscipline]) the list of children sub proxies
        """
        return self.proxy_disciplines

    def get_sub_proxy_disciplines(self, disc_list=None):
        '''
        Recursively returns all descendancy of sub proxies

        Arguments:
            disc_list (List[ProxyDiscipline]): current list of descendancy of sub proxies

        Returns:
            (List[ProxyDiscipline]): complete descendancy of sub proxies
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
         Property to obtain the ordered list of disciplines by default, for a ProxyDiscipline it is the order of
         sub proxy disciplines
        '''
        return self.proxy_disciplines

    def add_discipline(self, disc):
        '''
        Add a discipline to the children sub proxies and set self as father executor.

        Arguments:
            disc (ProxyDiscipline): discipline to add
        '''
        self.proxy_disciplines.append(disc)
        disc.set_father_executor(self)
        # self._check_if_duplicated_disc_names()

    def add_discipline_list(self, disc_list):
        '''
        Add a list of disciplines to the children sub proxies and set self as father executor.

        Arguments:
            disc_list (List[ProxyDiscipline]): disciplines to add
        '''
        for disc in disc_list:
            self.add_discipline(disc)

    def set_shared_namespaces_dependencies(self, data_dict):
        '''
        Set dependencies of shared inputs and outputs in ns_manager

        Arguments:
            data_dict (Dict[dict]): data_in or data_out
        '''
        shared_namespace_list = self.get_shared_namespace_list(
            data_dict)
        self.ee.ns_manager.add_dependencies_to_shared_namespace(
            self, shared_namespace_list)

    def add_variables(self, data_dict, io_type, clean_variables=True):
        '''
        Add dynamic inputs/outputs in ins_desc_in/ints_desc_out and remove old dynamic inputs/outputs

        Arguments:
            data_dict (Dict[dict]): new dynamic inputs/outputs
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
            clean_variables (bool): flag to remove old variables from data_in/data_out, inst_desc_in/inst_desc_out, datamanger
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

        Arguments:
            data_dict (Dict[dict]): new dynamic inputs
            clean_variables (bool): flag to remove old variables from data_in, inst_desc_in and datamanger
        '''
        self.add_variables(data_dict, self.IO_TYPE_IN,
                           clean_variables=clean_inputs)

    def add_outputs(self, data_dict, clean_outputs=True):
        '''
        Add dynamic outputs

        Arguments:
            data_dict (Dict[dict]): new dynamic outputs
            clean_variables (bool): flag to remove old variables from data_out, inst_desc_out and datamanger
        '''
        self.add_variables(data_dict, self.IO_TYPE_OUT,
                           clean_variables=clean_outputs)

    def clean_variables(self, var_name_list, io_type):
        '''
        Remove variables from data_in/data_out, inst_desc_in/inst_desc_out and datamanger

        Arguments:
            var_name_list (List[string]): variable names to clean
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
        '''
        for var_name in var_name_list:
            if io_type == self.IO_TYPE_IN:
                self.ee.dm.remove_keys(
                    self.disc_id, self.get_var_full_name(var_name, self.get_data_in()))

                del self._data_in[(var_name, self._io_ns_map_in[var_name])]
                del self._io_ns_map_in[var_name]

                del self.inst_desc_in[var_name]
            elif io_type == self.IO_TYPE_OUT:
                self.ee.dm.remove_keys(
                    self.disc_id, self.get_var_full_name(var_name, self.get_data_out()))

                del self._data_out[(var_name, self._io_ns_map_out[var_name])]
                del self._io_ns_map_out[var_name]

                del self.inst_desc_out[var_name]
            if var_name in self._structuring_variables:
                del self._structuring_variables[var_name]

    def update_default_value(self, var_name, io_type, new_default_value):
        '''
        Update DEFAULT and VALUE of var_name in data_io

        Arguments:
            var_name (string): variable names to clean
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
            new_default_value: value to update VALUE and DEFAULT with
        '''
        if var_name in self.get_data_io_dict(io_type):
            self.get_data_io_dict(
                io_type)[var_name][self.DEFAULT] = new_default_value
            self.get_data_io_dict(
                io_type)[var_name][self.VALUE] = new_default_value

    # -- Configure handling
    def configure(self):
        '''
        Configure the ProxyDiscipline
        '''
        # check if all config_dependency_disciplines are configured. If not no
        # need to try configuring the discipline because all is not ready for
        # it
        if self.check_configured_dependency_disciplines():
            self.set_numerical_parameters()

            if self.check_structuring_variables_changes():
                self.set_structuring_variables_values()

            self.setup_sos_disciplines()

            self.reload_io()

            # update discipline status to CONFIGURE
            self._update_status_dm(self.STATUS_CONFIGURE)

            self.set_configure_status(True)

    def __check_all_data_integrity(self):
        '''
         generic data integrity_check where we call different generic function to check integrity 
         + specific data integrity by discipline
        '''
        self.__generic_check_data_integrity()
        self.check_data_integrity()

    def check_data_integrity(self):
        pass

    def __generic_check_data_integrity(self):
        '''
        Generic check data integrity of the variables that you own ( the model origin of the variable is you)
        '''

        data_in_full_name = self.get_data_io_with_full_name(self.IO_TYPE_IN)
        for var_fullname in data_in_full_name:
            var_data_dict = self.dm.get_data(var_fullname)

            if var_data_dict['model_origin'] == self.disc_id:

                #                 check_integrity_msg = check_data_integrity_cls.check_variable_type_and_unit(var_data_dict)
                check_integrity_msg = self.check_data_integrity_cls.check_variable_value(
                    var_data_dict, self.ee.data_check_integrity)
                self.dm.set_data(
                    var_fullname, self.CHECK_INTEGRITY_MSG, check_integrity_msg)

    def set_numerical_parameters(self):
        '''
        Set numerical parameters of the ProxyDiscipline defined in the NUM_DESC_IN
        '''
        if self._data_in != {}:
            self.linearization_mode = self.get_sosdisc_inputs(
                'linearization_mode')

            self.update_reset_cache()

            self.update_reset_debug_mode()

    def update_reset_debug_mode(self):
        '''
        Update the reset_debug_mode boolean if debug mode has changed + logger 
        '''
        # Debug mode logging and recursive setting (priority to the parent)
        debug_mode = self.get_sosdisc_inputs(self.DEBUG_MODE)
        if debug_mode != self._structuring_variables[self.DEBUG_MODE]\
                and not (debug_mode == "" and self._structuring_variables[self.DEBUG_MODE] is None):  # not necessary on first config
            self._reset_debug_mode = True
            # logging
            if debug_mode != "":
                if debug_mode == "all":
                    for mode in self.AVAILABLE_DEBUG_MODE:
                        if mode not in ["", "all"]:
                            self.logger.info(
                                f'Discipline {self.sos_name} set to debug mode {mode}')
                else:
                    self.logger.info(
                        f'Discipline {self.sos_name} set to debug mode {debug_mode}')

    def update_reset_cache(self):
        '''
        Update the reset_cache boolean if cache type has changed
        '''
        cache_type = self.get_sosdisc_inputs(self.CACHE_TYPE)
        if cache_type != self._structuring_variables[self.CACHE_TYPE]:
            self._reset_cache = True
            self._set_children_cache = True

    def set_debug_mode_rec(self, debug_mode):
        """
        set debug mode recursively to children with priority to parent
        """
        for disc in self.proxy_disciplines:
            disc_in = disc.get_data_in()
            if ProxyDiscipline.DEBUG_MODE in disc_in:
                self.dm.set_data(self.get_var_full_name(
                    self.DEBUG_MODE, disc_in), self.VALUE, debug_mode, check_value=False)
                disc.set_debug_mode_rec(debug_mode)

    def setup_sos_disciplines(self):
        """
        Method to be overloaded to add dynamic inputs/outputs using add_inputs/add_outputs methods.
        If the value of an input X determines dynamic inputs/outputs generation, then the input X is structuring and the item 'structuring':True is needed in the DESC_IN
        DESC_IN = {'X': {'structuring':True}}
        """

        self.mdo_discipline_wrapp.setup_sos_disciplines(self)

    def set_dynamic_default_values(self, default_values_dict):
        """
        Method to set default value to a variable with short_name in a discipline when the default value varies with other input values
        i.e. a default array length depends on a number of years
        Arguments:
            default_values_dict (Dict[string]) : dict whose key is variable short name and value is the default value
        """
        disc_in = self.get_data_in()
        for short_key, default_value in default_values_dict.items():
            if short_key in disc_in:
                ns_key = self.get_var_full_name(short_key, disc_in)
                self.dm.no_check_default_variables.append(ns_key)
                self.dm.set_data(ns_key, self.DEFAULT, default_value, False)
            else:
                self.logger.info(
                    f'Try to set a default value for the variable {short_key} in {self.sos_name} which is not an input of this discipline ')

    # -- data handling section
    def reset_data(self):
        """
        Reset instance data attributes of the discipline to empty dicts.
        """
        self.inst_desc_in = {}
        self.inst_desc_out = {}
        self._data_in = {}
        self._data_out = {}
        self._io_ns_map_in = {}
        self._io_ns_map_out = {}

        self._structuring_variables = {}

    def get_data_in(self):
        """"
        _data_in getter
        #TODO: RENAME THIS METHOD OR ADD MODES 's'/'f'/'t' (short/full/tuple) as only the discipline dict and not subprocess is output
        """
        return {var_name: self._data_in[(var_name, id_ns)] for (var_name, id_ns) in self._io_ns_map_in.items()}

    def get_data_out(self):
        """
        _data_out getter
        #TODO: RENAME THIS METHOD OR ADD MODES 's'/'f'/'t' (short/full/tuple) as only the discipline dict and not subprocess is output
        """
        return {var_name: self._data_out[(var_name, id_ns)] for (var_name, id_ns) in self._io_ns_map_out.items()}

    def get_data_io_with_full_name(self, io_type, as_namespaced_tuple=False):
        """
        returns a version of the data_in/data_out of discipline with variable full names

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT

        Return:
            data_io_full_name (Dict[dict]): data_in/data_out with variable full names
        """
        # data_io_short_name = self.get_data_io_dict(io_type)
        #
        # if as_namespaced_tuple:
        #     def dict_key(v): return (
        #         v, id(data_io_short_name[v][self.NS_REFERENCE]))
        # else:
        #     def dict_key(v): return self.get_var_full_name(
        #         v, data_io_short_name)
        #
        # data_io_full_name = {dict_key(
        # var_name): value_dict for var_name, value_dict in
        # data_io_short_name.items()}

        if io_type == self.IO_TYPE_IN:
            if as_namespaced_tuple:
                return self._data_in
            else:
                return self.ns_tuples_to_full_name_keys(self._data_in)
        elif io_type == self.IO_TYPE_OUT:
            if as_namespaced_tuple:
                return self._data_out
            else:
                return self.ns_tuples_to_full_name_keys(self._data_out)
        else:
            raise ValueError('Unknown io type')

    def get_data_with_full_name(self, io_type, full_name, data_name=None):
        """
        Returns the field data_name in the data_in/data_out of a single variable based on its full_name. If data_name
        is None, returns the whole data dict of the variable.

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
            full_name (string): full name of the variable
            data_name (string): key of the data dict to get or None

        Return:
            (dict or Any) the data dict or its field [data_name]
        """
        data_io_full_name = self.get_data_io_with_full_name(io_type)

        if data_name is None:
            return data_io_full_name[full_name]
        else:
            return data_io_full_name[full_name][data_name]

    def get_ns_reference(self, visibility, namespace=None):
        '''
        Get namespace reference by consulting the namespace_manager

        Arguments:
            visibility (string): visibility to get local or shared namespace
            namespace (Namespace): namespace in case of shared visibility
        '''
        ns_manager = self.ee.ns_manager

        if visibility == self.LOCAL_VISIBILITY or visibility == self.INTERNAL_VISIBILITY:
            return ns_manager.get_local_namespace(self)

        elif visibility == self.SHARED_VISIBILITY:
            return ns_manager.get_shared_namespace(self, namespace)

    def apply_visibility_ns(self, io_type):
        '''
        Consult the namespace_manager to apply the namespace depending on the variable visibility

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
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

    def _prepare_data_dict(self, io_type, data_dict):  # =None):
        """
        Prepare the data_in/data_out with fields by default (and set _structuring_variables) for variables in data_dict.
        If data_dict is None, will prepare the current data_in/data_out.

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
            data_dict (Dict[dict]): the data dict to prepare
        """
        # if data_dict is None:
        #     data_dict = self.get_data_io_dict(io_type)
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
            if self.CHECK_INTEGRITY_MSG not in data_keys:
                curr_data[self.CHECK_INTEGRITY_MSG] = ''

            # initialize formula
            if self.FORMULA not in data_keys:
                curr_data[self.FORMULA] = None
            if self.IS_FORMULA not in data_keys:
                curr_data[self.IS_FORMULA] = False
            if self.IS_EVAL not in data_keys:
                curr_data[self.IS_EVAL] = False

        return data_dict

    def get_sosdisc_inputs(self, keys=None, in_dict=False, full_name_keys=False):
        """
        Accessor for the inputs values as a list or dict.

        Arguments:
            keys (List): the input short or full names list (depending on value of full_name_keys)
            in_dict (bool): if output format is dict
            full_name_keys (bool): if keys in args AND returned dictionary are full names or short names. Note that only
                                   True allows to query for variables of the subprocess as well as of the discipline itself.
        Returns:
            The inputs values list or dict
        """
        #TODO: refactor
        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            if full_name_keys:
                keys = list(self.get_data_io_with_full_name(
                    self.IO_TYPE_IN).keys())  # discipline and subprocess
            else:
                keys = list(self.get_data_in().keys())  # discipline only
            in_dict = True
        inputs = self._get_sosdisc_io(
            keys, io_type=self.IO_TYPE_IN, full_name_keys=full_name_keys)
        if in_dict:
            # return inputs in an dictionary
            return inputs
        else:
            # return inputs in an ordered tuple (default)
            if len(inputs) > 1:
                return list(inputs.values())
            else:
                return list(inputs.values())[0]

    def get_sosdisc_outputs(self, keys=None, in_dict=False, full_name_keys=False):
        """
        Accessor for the outputs values as a list or dict.

        Arguments:
            keys (List): the output short or full names list (depending on value of full_name_keys)
            in_dict (bool): if output format is dict
            full_name_keys (bool): if keys in args AND returned dictionary are full names or short names. Note that only
                                   True allows to query for variables of the subprocess as well as of the discipline itself.
        Returns:
            The outputs values list or dict
        """
        #TODO: refactor
        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            if full_name_keys:
                keys = list(self.get_data_io_with_full_name(
                    self.IO_TYPE_OUT).keys())  # discipline and subprocess
            else:
                keys = list(self.get_data_out().keys())  # discipline only
            # keys = [d[self.VAR_NAME] for d in self.get_data_out().values()]
            in_dict = True
        outputs = self._get_sosdisc_io(
            keys, io_type=self.IO_TYPE_OUT, full_name_keys=full_name_keys)
        if in_dict:
            # return outputs in an dictionary
            return outputs
        else:
            # return outputs in an ordered tuple (default)
            if len(outputs) > 1:
                return list(outputs.values())
            else:
                return list(outputs.values())[0]

    def _get_sosdisc_io(self, keys, io_type, full_name_keys=False):
        """
        Generic method to retrieve sos inputs and outputs

        Arguments:
            keys (List[String]): the data names list in short or full names (depending on value of full_name_keys)
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
            full_name_keys: if keys in args and returned dict are full names. Note that only True allows to query for
                            variables of the subprocess as well as of the discipline itself.
        Returns:
            dict of keys values
        Raises:
            Exception if query key is not in the data manager
        """
        #TODO: refactor
        if isinstance(keys, str):
            keys = [keys]

        if full_name_keys:
            query_keys = keys
        else:
            query_keys = self._convert_list_of_keys_to_namespace_name(
                keys, io_type)

        values_dict = {}
        for key, q_key in zip(keys, query_keys):
            if q_key not in self.dm.data_id_map:
                print('toto')
                raise Exception(
                    f'The key {q_key} for the discipline {self.get_disc_full_name()} is missing in the data manager')
            # get data in local_data during run or linearize steps
            # #TODO: this should not be possible in command line mode, is it possible in the GUI?
            elif self.status in [self.STATUS_RUNNING, self.STATUS_LINEARIZE]:
                values_dict[key] = self.mdo_discipline_wrapp.mdo_discipline.local_data[q_key]
            # get data in data manager during configure step
            else:
                values_dict[key] = self.dm.get_value(q_key)
        return values_dict

    # def _get_sosdisc_io(self, keys, io_type, full_name=False):
    #     """
    #     Generic method to retrieve discipline inputs and outputs
    #
    #     Arguments:
    #         keys (List[string]): the output short names list
    #         io_type (string): IO_TYPE_IN or IO_TYPE_OUT
    #         full_name (bool): True if returned keys are full names, False for short names
    #
    #     Return:
    #         values_dict (dict): dict of variable keys and values
    #     """
    #
    #     # convert local key names to namespaced ones
    #     if isinstance(keys, str):
    #         keys = [keys]
    #     namespaced_keys_dict = {key: namespaced_key for key, namespaced_key in zip(
    #         keys, self._convert_list_of_keys_to_namespace_name(keys, io_type))}
    #
    #     values_dict = {}
    #
    #     for key, namespaced_key in namespaced_keys_dict.items():
    #         # new_key can be key or namespaced_key according to full_name value
    #         new_key = full_name * namespaced_key + (1 - full_name) * key
    #         if namespaced_key not in self.dm.data_id_map:
    #             raise Exception(
    #                 f'The key {namespaced_key} for the discipline {self.get_disc_full_name()} is missing in the data manager')
    #         # get data in local_data during run or linearize steps
    #         elif self.status in [self.STATUS_RUNNING, self.STATUS_LINEARIZE]:
    #             values_dict[new_key] = self.mdo_discipline_wrapp.mdo_discipline.local_data[namespaced_key]
    #         # get data in data manager during configure step
    #         else:
    #             values_dict[new_key] = self.dm.get_value(namespaced_key)
    #
    #     return values_dict

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

    #     def get_input_data_for_gems(self):
    #         '''
    #         Get input_data for linearize ProxyDiscipline
    #         '''
    #         input_data = {}
    #         input_data_names = self.input_grammar.get_data_names()
    #         if len(input_data_names) > 0:
    #
    #             for data_name in input_data_names:
    #                 input_data[data_name] = self.ee.dm.get_value(data_name)
    #
    #         return input_data

    def _update_type_metadata(self):
        '''
        Update metadata of values not supported by GEMS (for cases where the data has been converted by the coupling)
        '''
        disc_in = self.get_data_in()
        for var_name in disc_in.keys():
            var_f_name = self.get_var_full_name(var_name, disc_in)
            var_type = self.dm.get_data(var_f_name, self.VAR_TYPE_ID)
            if var_type in self.NEW_VAR_TYPE:
                if self.dm.get_data(var_f_name, self.TYPE_METADATA) is not None:
                    disc_in[var_name][self.TYPE_METADATA] = self.dm.get_data(
                        var_f_name, self.TYPE_METADATA)

        disc_out = self.get_data_out()
        for var_name in disc_out.keys():
            var_f_name = self.get_var_full_name(var_name, disc_out)
            var_type = self.dm.get_data(var_f_name, self.VAR_TYPE_ID)
            if var_type in self.NEW_VAR_TYPE:
                if self.dm.get_data(var_f_name, self.TYPE_METADATA) is not None:
                    disc_out[var_name][self.TYPE_METADATA] = self.dm.get_data(
                        var_f_name, self.TYPE_METADATA)

    def _update_study_ns_in_varname(self, names):
        '''
        Updates the study name in the variable input names.

        Arguments:
            names (List[string]): names to update
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

    def update_meta_data_out(self, new_data_dict):
        """
        update meta data of _data_out and DESC_OUT

        Arguments:
            new_data_dict (Dict[dict]): contains the metadata to be updated
                                        in format: {'variable_name' : {'meta_data_name' : 'meta_data_value',...}....}
        """
        disc_out = self.get_data_out()
        for key in new_data_dict.keys():
            for meta_data in new_data_dict[key].keys():
                disc_out[key][meta_data] = new_data_dict[key][meta_data]
                if meta_data in self.DESC_OUT[key].keys():
                    self.DESC_OUT[key][meta_data] = new_data_dict[key][meta_data]

    def clean_dm_from_disc(self):
        """
        Clean ProxyDiscipline in datamanager's disciplines_dict and data_in/data_out keys
        """
        self.dm.clean_from_disc(self.disc_id)

    def _set_dm_disc_info(self):
        """
        Set info of the ProxyDiscipline in datamanager
        """
        disc_ns_name = self.get_disc_full_name()
        disc_dict_info = {}
        disc_dict_info['reference'] = self
        disc_dict_info['classname'] = self.__class__.__name__
#         disc_dict_info['model_name'] = self.__module__.split('.')[-2]
        disc_dict_info['model_name_full_path'] = self.get_module()
        disc_dict_info['treeview_order'] = 'no'
        disc_dict_info[self.NS_REFERENCE] = self.ee.ns_manager.get_local_namespace(
            self)
        self.disc_id = self.dm.update_disciplines_dict(
            self.disc_id, disc_dict_info, disc_ns_name)

    def _set_dm_cache_map(self):
        '''
        Update cache_map dict in DM with cache and its children recursively
        '''

        mdo_discipline = self.mdo_discipline_wrapp.mdo_discipline if self.mdo_discipline_wrapp is not None else None
        if mdo_discipline is not None:
            self._store_cache_with_hashed_uid(mdo_discipline)
        # store children cache recursively
        for disc in self.proxy_disciplines:
            disc._set_dm_cache_map()

    def _store_cache_with_hashed_uid(self, disc):
        '''
        Generate hashed uid and store cache in DM
        '''
        if disc.cache is not None:
            disc_info_list = self.get_disc_info_list_for_hashed_uid(disc)
            # store cache in DM map
            self.dm.fill_cache_map(disc_info_list, disc)

    def get_disc_info_list_for_hashed_uid(self, disc):
        full_name = self.get_disc_full_name().split(self.ee.study_name)[-1]
        class_name = disc.__class__.__name__
        data_io_string = self.get_single_data_io_string_for_disc_uid(disc)

        # set disc infos string list with full name, class name and anonimated
        # i/o for hashed uid generation
        return [full_name, class_name, data_io_string]

    def get_cache_map_hashed_uid(self, disc):
        disc_info_list = self.get_disc_info_list_for_hashed_uid(disc)
        hashed_uid = self.dm.generate_hashed_uid(disc_info_list)
        return hashed_uid

    def get_var_full_name(self, var_name, disc_dict):
        '''
        Get namespaced variable from namespace and var_name in disc_dict
        '''
        ns_reference = disc_dict[var_name][self.NS_REFERENCE]
        complete_var_name = disc_dict[var_name][self.VAR_NAME]
        var_f_name = self.ee.ns_manager.compose_ns(
            [ns_reference.value, complete_var_name])
        return var_f_name

    def ns_tuples_to_full_name_keys(self, in_dict):
        """
        Converts the keys of the input dictionary from tuples (var_short_name, id(ns_ref)) to strings var_full_name.

        Arguments:
            in_dict (dict[Any]): the input dictionary whose keys are tuples (var_short_name, id(ns_ref))

        Returns:
            dict[Any]: the dictionary with same values and full name keys
        """
        return {self.ee.ns_manager.ns_tuple_to_full_name(var_ns_tuple): value for var_ns_tuple, value in in_dict.items()}

    def update_from_dm(self):
        """
        Update all disciplines with datamanager information
        """

        self.__check_all_data_integrity()

        disc_in = self.get_data_in()
        for var_name in disc_in.keys():

            try:
                var_f_name = self.get_var_full_name(var_name, disc_in)
                default_val = self.dm.data_dict[self.dm.get_data_id(
                    var_f_name)][self.DEFAULT]
            except:
                var_f_name = self.get_var_full_name(var_name, disc_in)
            if self.dm.get_value(var_f_name) is None and default_val is not None:
                disc_in[var_name][self.VALUE] = default_val
            else:
                # update from dm for all proxy_disciplines to load all data
                disc_in[var_name][self.VALUE] = self.dm.get_value(
                    var_f_name)
        # -- update sub-disciplines
        for discipline in self.proxy_disciplines:
            discipline.update_from_dm()

    # -- Ids and namespace handling
    def get_disc_full_name(self):
        '''
        Return: (string) the discipline name with full namespace
        '''
        return self.ee.ns_manager.get_local_namespace_value(self)

    def get_disc_id_from_namespace(self):
        """
        Return: (string) the discipline id
        """
        return self.ee.dm.get_discipline_ids_list(self.get_disc_full_name())

    def get_single_data_io_string_for_disc_uid(self, disc):
        '''
        Return: (List[string]) of anonimated input and output keys for serialisation purpose
        '''

        input_list_anonimated = [key.split(
            self.ee.study_name, 1)[-1] for key in disc.get_input_data_names()]
        input_list_anonimated.sort()
        output_list_anonimated = [key.split(
            self.ee.study_name, 1)[-1] for key in disc.get_output_data_names()]
        output_list_anonimated.sort()
        input_list_anonimated.extend(output_list_anonimated)

        return ''.join(input_list_anonimated)

    def _convert_list_of_keys_to_namespace_name(self, keys, io_type):
        """
        Convert a list of keys to namespace name (see _convert_to_namespace_name).

        Arguments:
            keys (List[string]): list of keys to convert
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT

        Return:
            variables (list[string]): the list of varaible namespace name
        """
        # Refactor  variables keys with namespace
        if isinstance(keys, list):
            variables = [self._convert_to_namespace_name(
                key, io_type) for key in keys]
        else:
            variables = [self._convert_to_namespace_name(
                keys, io_type)]
        return variables

    def _convert_to_namespace_name(self, key, io_type):
        ''' Convert to namespace with coupling_namespace management
            Using a key (variables name) and reference_data (yaml in or out),
            build the corresponding namespaced key using the visibility property

            Arguments:
                key (string): variable name

            Return:
                (string) the variable namespace name
        '''

        # Refactor  variables keys with namespace
        result = self.ee.ns_manager.get_namespaced_variable(
            self, key, io_type)

        return result

    # -- status handling section
    def _update_status_dm(self, status):
        """
        Update discipline _status and status in data manager.

        Arguments:
            status (string): the status to update
        """
        # Avoid unnecessary call to status property (which can trigger event in
        # case of change)
        if self._status != status:
            self._status = status
            if self.mdo_discipline_wrapp is not None and self.mdo_discipline_wrapp.mdo_discipline is not None:
                self.mdo_discipline_wrapp.mdo_discipline.status = status

        # Force update into discipline_dict (GEMS can change status but cannot update the
        # discipline_dict
        self.dm.disciplines_dict[self.disc_id]['status'] = status

    def _update_status_recursive(self, status):
        """
        Update discipline _status and status in data manager recursively for self and descendancy of sub proxies and in
        data manager.

        Arguments:
            status (string): the status to update
        """
        # keep reference branch status to 'REFERENCE'
        self._update_status_dm(status)
        for disc in self.proxy_disciplines:
            disc._update_status_recursive(status)

    def set_status_from_mdo_discipline(self):
        """
        Update status of self and children sub proxies by retreiving the status of the GEMSEO objects.

        """
        for proxy_discipline in self.proxy_disciplines:
            proxy_discipline.set_status_from_mdo_discipline()
        self.status = self.get_status_after_configure()

    def get_status_after_configure(self):
        if self.mdo_discipline_wrapp.mdo_discipline is not None:
            return self.mdo_discipline_wrapp.mdo_discipline.status
        else:
            return self._status

    def add_status_observer(self, observer):
        '''
        Observer has to be set before execution (and prepare_execution) and the mdo_discipline does not exist. 
        We store observers in self.status_observers and add it to the mdodiscipline when it ies instanciated in prepare_execution
        '''
        if self.mdo_discipline_wrapp is not None and self.mdo_discipline_wrapp.mdo_discipline is not None:
            self.mdo_discipline_wrapp.mdo_discipline.add_status_observer(
                observer)

        if observer not in self.status_observers:
            self.status_observers.append(observer)

    def remove_status_observer(self, observer):
        '''
        Remove the observer from the status_observers list
        And normally the mdodiscipline has already been instanciated and we can remove it. 
        If not the case the mdodiscipline does not exist such as the observer
        '''
        if observer in self.status_observers:
            self.status_observers.remove(observer)
        if self.mdo_discipline_wrapp is not None and self.mdo_discipline_wrapp.mdo_discipline is not None:
            self.mdo_discipline_wrapp.mdo_discipline.remove_status_observer(
                observer)

    def _check_status_before_run(self):
        """
        Check discipline status is ok before run and throw ValueError otherwise.
        """
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
        """
        Maturity setter

        Arguments:
            maturity (string or dict or None): maturity to set
            maturity_dict (bool): whether the maturity is a dict
        """
        if maturity is None or maturity in self.possible_maturities or maturity_dict:
            self._maturity = maturity
        else:
            raise Exception(
                f'Unkown maturity {maturity} for discipline {self.sos_name}')

    def get_maturity(self):
        '''
        Get the maturity of the ProxyDiscipline (a discipline does not have any subdisciplines, only a coupling has)
        '''
        if hasattr(self, '_maturity'):
            return self._maturity
        elif hasattr(self.mdo_discipline_wrapp, 'wrapper'):
            if hasattr(self.mdo_discipline_wrapp.wrapper, '_maturity'):
                return self.mdo_discipline_wrapp.wrapper._maturity
            else:
                return ''
        return ''

    def _build_dynamic_DESC_IN(self):
        pass

    def get_chart_filter_list(self):
        """
        Return a list of ChartFilter instance base on the inherited class post processing filtering capabilities

        Returns: List[ChartFilter]
        """
        if self.mdo_discipline_wrapp is not None and self.mdo_discipline_wrapp.wrapper is not None:
            return self.mdo_discipline_wrapp.wrapper.get_chart_filter_list()
        else:
            return []

    def get_post_processing_list(self, filters=None):
        """
        Return a list of post processing instance using the ChartFilter list given as parameter

        Arguments:
            filters: filters to apply during post processing making

        Returns:
            post processing instance list
        """
        if self.mdo_discipline_wrapp is not None and self.mdo_discipline_wrapp.wrapper is not None:
            return self.mdo_discipline_wrapp.wrapper.get_post_processing_list(self, filters)

        else:
            return []

    def set_configure_status(self, is_configured):
        """
        Set boolean is_configured which indicates if the discipline has been configured
        to avoid several configuration in a multi-level process and save time
        """

        self._is_configured = is_configured

    def get_configure_status(self):
        """
        Get boolean is_configured which indicates if the discipline has been configured
        to avoid several configuration in a multi-level process and save time
        """

        if hasattr(self, '_is_configured'):
            return self._is_configured
        else:
            return ''

    def is_configured(self):
        '''
        Return False if discipline needs to be configured, True if not
        '''
        return self.get_configure_status() and not self.check_structuring_variables_changes() and self.check_configured_dependency_disciplines()

    def check_configured_dependency_disciplines(self):
        '''
        Check if config_dependency_disciplines are configured to know if i am configured
        Be careful using this capability to avoid endless loop of configuration 
        '''
        return all([disc.is_configured() for disc in self.config_dependency_disciplines])

    def add_disc_to_config_dependency_disciplines(self, disc):
        '''
        Add a discipline to config_dependency_disciplines 
        Be careful to endless configuraiton loop (small loops are checked but not with more than two disciplines)
        Do not add twice the same dsicipline
        '''
        if disc == self:
            error_msg = f'Not possible to add self in the config_dependency_list for disc : {disc.get_disc_full_name()}'
            self.logger.error(error_msg)
            raise Exception(error_msg)

        if self in disc.config_dependency_disciplines:
            error_msg = f'The discipline {disc.get_disc_full_name()} has already {self.get_disc_full_name()} in its config_dependency_list, it is not possible to add the discipline in config_dependency_list of myself'
            self.logger.error(error_msg)
            raise Exception(error_msg)

        if disc not in self.__config_dependency_disciplines:
            self.__config_dependency_disciplines.append(disc)
            disc.add_dependent_disciplines(self)

    def delete_disc_in_config_dependency_disciplines(self, disc):

        self.__config_dependency_disciplines.remove(disc)

    def add_dependent_disciplines(self, disc):

        self.__config_dependent_disciplines.append(disc)

    def clean_config_dependency_disciplines_of_dependent_disciplines(self):

        for disc in self.__config_dependent_disciplines:
            disc.delete_disc_in_config_dependency_disciplines(self)

    def add_disc_list_to_config_dependency_disciplines(self, disc_list):
        '''
        Add a list to children_list 
        '''
        for disc in disc_list:
            self.add_disc_to_config_dependency_disciplines(disc)

    def get_disciplines_to_configure(self):
        '''
        Get sub disciplines list to configure according to their is_configured method (coupling, eval, etc.)
        '''
        disc_to_configure = []
        for disc in self.proxy_disciplines:
            if not disc.is_configured():
                disc_to_configure.append(disc)
        return disc_to_configure

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
        disc_in = self.get_data_in()
        for struct_var in list(self._structuring_variables.keys()):
            if struct_var in disc_in:
                self._structuring_variables[struct_var] = deepcopy(
                    self.get_sosdisc_inputs(struct_var))

    # ----------------------------------------------------
    # ----------------------------------------------------
    #  METHODS TO DEBUG DISCIPLINE
    # ----------------------------------------------------
    # ----------------------------------------------------

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

    def clean(self):
        """
        This method cleans a sos_discipline;
        In the case of a "simple" discipline, it removes the discipline from
        its father builder and from the factory sos_discipline. This is achieved
        by the method remove_sos_discipline of the factory
        """
        self.father_builder.remove_discipline(self)
        self.clean_dm_from_disc()
        self.ee.ns_manager.remove_dependencies_after_disc_deletion(
            self, self.disc_id)
        self.ee.factory.remove_sos_discipline(self)
        self.clean_config_dependency_disciplines_of_dependent_disciplines()

    def set_wrapper_attributes(self, wrapper):
        """
        set the attribute ".attributes" of wrapper which is used to provide the wrapper with information that is
        figured out at configuration time but needed at runtime. the input and output full name map allow the wrappers
        to work with short names whereas the GEMSEO objects use variable full names in their data structures.
        """
        input_full_name_map, output_full_name_map = self.create_io_full_name_map()
        wrapper.attributes = {
            'input_full_name_map': input_full_name_map,
            'output_full_name_map': output_full_name_map
        }

    # def set_discipline_attributes(self, discipline):
    #     """ set the attribute attributes of mdo_discipline --> not needed if using SoSMDODisciplineDriver
    #     """
    #     pass

    def create_io_full_name_map(self):
        """
        Create an io_full_name_map as wel ass input_full_name_map and output_full_name_map for its sos_wrapp

        Return:
            input_full_name_map (Dict[Str]): dict whose keys are input short names and values are input full names
            output_full_name_map (Dict[Str]): dict whose keys are output short names and values are output full names
        """

        return {key: self.ee.ns_manager.ns_tuple_to_full_name((key, value)) for key, value in self._io_ns_map_in.items()},\
               {key: self.ee.ns_manager.ns_tuple_to_full_name(
                   (key, value)) for key, value in self._io_ns_map_out.items()}

    def get_module(self):
        '''
        Obtain the module of the wrapper if it exists. useful for postprocessing factory and treenode
        '''
        if self.mdo_discipline_wrapp is not None and self.mdo_discipline_wrapp.wrapper is not None:
            disc_module = self.mdo_discipline_wrapp.wrapper.__module__

        else:
            # for discipline not associated to wrapper (proxycoupling for
            # example)
            disc_module = self.__module__
        # return the replace sostrades_core just for documentation (linked
        # ontology is the one from integration)
        return disc_module.replace(
            'sostrades_core', 'sos_trades_core')
    # useful for debugging

    def display_proxy_subtree(self, callback=None):
        proxy_subtree = []
        self.get_proxy_subtree_rec(proxy_subtree, 0, callback)
        return '\n'.join(proxy_subtree)

    def get_proxy_subtree_rec(self, proxy_subtree, indent=0, callback=None):
        callback_string = ' [' + str(callback(self)) + \
            ']' if callback is not None else ''
        proxy_subtree.append('    ' * indent + '|_ ' + self.ee.ns_manager.get_local_namespace_value(self)
                             + '  (' + self.__class__.__name__ + ')' + callback_string)
        for disc in self.proxy_disciplines:
            disc.get_proxy_subtree_rec(proxy_subtree, indent + 1, callback)
