'''
Copyright 2022 Airbus SAS
Modifications on 2023/02/23-2024/06/28 Copyright 2023 Capgemini

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

from __future__ import annotations

import contextlib
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from gemseo.core.discipline.discipline import Discipline
from gemseo.core.discipline.io import IO
from gemseo.core.execution_status import ExecutionStatus
from gemseo.core.process_discipline import ProcessDiscipline
from numpy import bool_ as np_bool
from numpy import complex128 as np_complex128
from numpy import float32 as np_float32
from numpy import float64 as np_float64
from numpy import int32 as np_int32
from numpy import int64 as np_int64
from numpy import ndarray
from pandas import DataFrame

from sostrades_core.execution_engine.discipline_wrapp import DisciplineWrapp
from sostrades_core.execution_engine.sos_discipline import SoSDiscipline
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.check_data_integrity.check_data_integrity import CheckDataIntegrity
from sostrades_core.tools.compare_data_manager_tooling import dict_are_equal

if TYPE_CHECKING:
    import logging


class ProxyDisciplineException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'


class ProxyDiscipline:
    """
    **ProxyDiscipline** is a class proxy for a  discipline on the SoSTrades side.

    It contains the information and methonds necessary for i/o configuration (static or dynamic).

    Leaves of the process tree are direct instances of ProxyDiscipline. Other nodes are instances that inherit from
    ProxyDiscipline (e.g. ProxyCoupling).

    An instance of ProxyDiscipline is in one-to-one aggregation with an instance of DisciplineWrapp, which allows the
    use of different wrapping modes to provide the model run.

    During the prepare_execution step, the ProxyDiscipline coordinates the instantiation of the GEMSEO objects that
    manage the model run.

    Attributes:
        discipline_wrapp (DisciplineWrapp): aggregated object that references the wrapper and GEMSEO discipline

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
    IO_TYPE = SoSWrapp.IO_TYPE
    IO_TYPE_IN = SoSWrapp.IO_TYPE_IN
    IO_TYPE_OUT = SoSWrapp.IO_TYPE_OUT
    TYPE = SoSWrapp.TYPE
    SUBTYPE = SoSWrapp.SUBTYPE
    COUPLING = SoSWrapp.COUPLING
    VISIBILITY = SoSWrapp.VISIBILITY
    LOCAL_VISIBILITY = SoSWrapp.LOCAL_VISIBILITY
    INTERNAL_VISIBILITY = SoSWrapp.INTERNAL_VISIBILITY
    SHARED_VISIBILITY = SoSWrapp.SHARED_VISIBILITY
    AVAILABLE_VISIBILITIES = [LOCAL_VISIBILITY, INTERNAL_VISIBILITY, SHARED_VISIBILITY]
    NAMESPACE = SoSWrapp.NAMESPACE
    NS_REFERENCE = 'ns_reference'
    REFERENCE = 'reference'
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
    RUN_NEEDED = SoSWrapp.RUN_NEEDED
    SUBTYPE: SoSWrapp.SUBTYPE
    META_INPUT = 'meta_input'
    OPTIONAL = 'optional'
    ORIGIN = 'model_origin'
    MODEL_NAME_FULL_PATH = 'model_name_full_path'
    HEADERS = 'headers'
    COMPOSED_OF = 'composed_of'
    DISCIPLINES_DEPENDENCIES = 'disciplines_dependencies'
    VAR_NAME = SoSWrapp.VAR_NAME
    VISIBLE = SoSWrapp.VISIBLE
    CACHE_TYPE = 'cache_type'
    CACHE_FILE_PATH = 'cache_file_path'
    FORMULA = 'formula'
    IS_FORMULA = 'is_formula'
    IS_EVAL = 'is_eval'
    CHECK_INTEGRITY_MSG = 'check_integrity_msg'
    VARIABLE_KEY = 'variable_key'  # key for ontology
    SIZE_MO = 'size_mo'  # size of a data
    DISPLAY_NAME = 'display_name'
    DATA_TO_CHECK = [TYPE, UNIT, RANGE, POSSIBLE_VALUES, USER_LEVEL]
    NO_UNIT_TYPES = ['bool', 'string', 'string_list']
    # Dict  ex: {'ColumnName': (column_data_type, column_data_range,
    # column_editable)}
    DATAFRAME_DESCRIPTOR = SoSWrapp.DATAFRAME_DESCRIPTOR
    DYNAMIC_DATAFRAME_COLUMNS = SoSWrapp.DYNAMIC_DATAFRAME_COLUMNS
    DATAFRAME_EDITION_LOCKED = SoSWrapp.DATAFRAME_EDITION_LOCKED
    #
    DF_EXCLUDED_COLUMNS = 'dataframe_excluded_columns'
    DEFAULT_EXCLUDED_COLUMNS = ['year', 'years']
    DISCIPLINES_FULL_PATH_LIST = 'discipline_full_path_list'

    # -- Variable types information section
    VAR_TYPE_ID = 'type'
    # complex can also be a type if we use complex step
    INT_MAP = (int, np_int32, np_int64, np_complex128)
    FLOAT_MAP = (float, np_float64, np_float32, np_complex128)
    BOOL_MAP = (bool, np_bool)
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
        'bool': BOOL_MAP,
        'list': list,
        PROC_BUILDER_MODAL: dict,
    }
    VAR_TYPE_GEMS = ['int', 'array', 'float_list', 'int_list']
    STANDARD_TYPES = [int, float, np_int32, np_int64, np_float64, bool]
    NEW_VAR_TYPE = ['dict', 'dataframe', 'string_list', 'string', 'float', 'int', 'list']

    UNSUPPORTED_GEMSEO_TYPES = []
    for _var_type in VAR_TYPE_MAP:
        if _var_type not in VAR_TYPE_GEMS and _var_type not in NEW_VAR_TYPE:
            # Fixing PERF401 would require heavy refactoring
            UNSUPPORTED_GEMSEO_TYPES.append(_var_type)  # noqa: PERF401

    # # Warning : We cannot put string_list into dict, all other types inside a dict are possiblr with the type dict
    # # df_dict = dict , string_dict = dict, list_dict = dict
    TYPE_METADATA = "type_metadata"

    POS_IN_MODE = ['value', 'list', 'dict']

    DEBUG_MODE = SoSDiscipline.DEBUG_MODE
    LINEARIZATION_MODE = SoSDiscipline.LINEARIZATION_MODE
    RESIDUAL_VARIABLES = SoSDiscipline.RESIDUAL_VARIABLES
    RUN_SOLVE_RESIDUALS = SoSDiscipline.RUN_SOLVE_RESIDUALS
    AVAILABLE_DEBUG_MODE = ["", "nan", "input_change", "min_max_couplings", "all"]

    # -- status section

    # -- Maturity section
    possible_maturities = ['Fake', 'Research', 'Official', 'Official Validated']
    dict_maturity_ref = dict(zip(possible_maturities, [0] * len(possible_maturities)))

    NUM_DESC_IN = {
        LINEARIZATION_MODE: {
            TYPE: 'string',
            DEFAULT: Discipline.ApproximationMode.FINITE_DIFFERENCES,
            POSSIBLE_VALUES: list(Discipline.LinearizationMode),
            NUMERICAL: True,
            STRUCTURING: True,
        },
        CACHE_TYPE: {
            TYPE: 'string',
            DEFAULT: Discipline.CacheType.NONE,
            POSSIBLE_VALUES: [Discipline.CacheType.NONE, Discipline.CacheType.SIMPLE],
            # [MDOChain.CacheType.NONE, Discipline.SIMPLE_CACHE, Discipline.HDF5_CACHE, Discipline.MEMORY_FULL_CACHE]
            NUMERICAL: True,
            STRUCTURING: True,
        },
        CACHE_FILE_PATH: {TYPE: 'string', DEFAULT: '', NUMERICAL: True, OPTIONAL: True, STRUCTURING: True},
        DEBUG_MODE: {
            TYPE: 'string',
            DEFAULT: '',
            POSSIBLE_VALUES: list(AVAILABLE_DEBUG_MODE),
            NUMERICAL: True,
            STRUCTURING: True,
        },
        RESIDUAL_VARIABLES: {TYPE: 'dict', DEFAULT: {}, SUBTYPE: {'dict': 'string'}, NUMERICAL: True},
        RUN_SOLVE_RESIDUALS: {TYPE: 'bool', DEFAULT: False, NUMERICAL: True},
    }

    # GLOBAL GEMSEO SETTINGS
    # -- grammars
    SOS_GRAMMAR_TYPE = "SoSSimpleGrammar"
    ProcessDiscipline.default_grammar_type = SOS_GRAMMAR_TYPE
    # -- status
    STATUS_PENDING = "PENDING"
    STATUS_DONE = ExecutionStatus.Status.DONE
    STATUS_RUNNING = ExecutionStatus.Status.RUNNING
    STATUS_FAILED = ExecutionStatus.Status.FAILED
    STATUS_CONFIGURE = 'CONFIGURE'
    STATUS_LINEARIZE = ExecutionStatus.Status.LINEARIZING

    EE_PATH = 'sostrades_core.execution_engine'

    io: IO
    """The GEMSEO object that contains the inputs/outputs of a discipline.

    Used by GEMSEO to create the coupling structure.
    """

    def __init__(self, sos_name, ee, cls_builder=None, associated_namespaces=None):
        """
        Constructor

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
            cls_builder (Class): class constructor of the user-defined wrapper (or None)
            associated_namespaces(List[string]): list containing ns ids ['name__value'] for namespaces associated to builder
        """
        # Must assign logger before calling create_discipline_wrap
        self.logger = ee.logger.getChild(self.__class__.__name__)
        # Enable not a number check in execution result and jacobian result
        # Be carreful that impact greatly calculation performances
        self.discipline_wrapp = None
        self.stored_cache = None
        self.create_discipline_wrap(name=sos_name, wrapper=cls_builder, wrapping_mode='SoSTrades', logger=self.logger)
        self._reload(sos_name, ee, associated_namespaces=associated_namespaces)

        self.model = None
        self.__father_builder = None
        self.father_executor: ProxyDiscipline | None = None
        self.cls = cls_builder
        self.io = None

    @property
    def configurator(self):
        """
        Property that is None when the discipline is self-configured and stores a reference to the configurator
        discipline otherwise.
        """
        return self.__configurator

    @configurator.setter
    def configurator(self, disc):
        """Configurator discipline setter."""
        if disc is self:
            self.__configurator = None
        else:
            self.__configurator = disc

    def set_father_executor(self, father_executor):  #: "ProxyDiscipline"):
        """
        set father executor

        Arguments:
            father_executor (ProxyDiscipline): proxy that orchestrates the execution of this proxy discipline (e.g. coupling)
        """
        self.father_executor = father_executor

    def _add_optional_shared_ns(self):
        """Adds the shared namespaces that have a default value depending on the Proxy type. To be overload in subclasses."""

    def _reload(
        self, sos_name, ee, associated_namespaces=None
    ):  #: str, ee: "ExecutionEngine", associated_namespaces: Union[list[str], None]  = None):
        """
        Reload ProxyDiscipline attributes and set is_sos_coupling.

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
            associated_namespaces(List[string]): list containing ns ids ['name__value'] for namespaces associated to builder
            logger (logging.Logger): logger to use
        """
        self.logger = ee.logger.getChild(self.__class__.__name__)
        self.proxy_disciplines: list[ProxyDiscipline] = []
        # : list of outputs that shall be null, to be considered as residuals
        self.residual_variables = {}
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
        self._add_optional_shared_ns()
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
        self._reset_linearization_mode: bool = False

        # -- disciplinary data attributes
        self.inst_desc_in = None  # desc_in of instance used to add dynamic inputs
        self.inst_desc_out = None  # desc_out of instance used to add dynamic outputs

        self._data_in = None
        self._data_out = None
        self._io_ns_map_in = None
        self._io_ns_map_out = None  # used by ProxyCoupling, ProxyDriverEvaluator
        self._structuring_variables = None
        self.reset_data()

        self._non_structuring_variables = None
        self.__all_input_structuring = False

        # -- Maturity attribute
        self._maturity = self.get_maturity()

        # Add the discipline in the dm and get its unique disc_id (was in the
        # configure)
        self._set_dm_disc_info()

        # Instantiate check_data_integrity class to check data after dm save
        self.check_data_integrity_cls = CheckDataIntegrity(self.__class__, self.dm)
        # update discipline status to CONFIGURE
        self._update_status_dm(self.STATUS_CONFIGURE)
        self.__configurator: ProxyDiscipline | None = None

    @property
    def name(self) -> str:
        """Return the full proxy name.

        Used by GEMSEO.
        """
        return self.get_disc_full_name()

    def create_discipline_wrap(self, name: str, wrapper, wrapping_mode: str, logger: logging.Logger):
        """
        creation of discipline_wrapp by the proxy
        To be overloaded by proxy without DisciplineWrapp (eg scatter...)
        """
        self.discipline_wrapp = DisciplineWrapp(
            name=name, logger=logger.getChild("DisciplineWrapp"), wrapper=wrapper, wrapping_mode=wrapping_mode
        )
        # self.assign_proxy_to_wrapper()
        # NB: this above is is problematic because made before dm assignation in ProxyDiscipline._reload, but it is also
        # unnecessary as long as no wrapper configuration actions are demanded BEFORE first proxy configuration.

    @property
    def status(self):  # type: (...) -> str
        """The status of the discipline, to be retrieved from the GEMSEO object after configuration."""
        if self._status != self.STATUS_CONFIGURE:
            return self.get_status_after_configure()
        return self.STATUS_CONFIGURE

    @property
    def father_builder(self):
        """The SoSBuilder that have built the discipline , Proxycoupling has no father_builder"""
        try:
            return self.__father_builder
        except:
            return None

    @father_builder.setter
    def father_builder(self, builder):
        """Setter of father_builder"""
        self.__father_builder = builder

    @property
    def config_dependency_disciplines(self):  # type: (...) -> str
        """The config_dependency_disciplines list which represents the list of disciplines that must be configured before you configure"""
        return self.__config_dependency_disciplines

    @property
    def config_dependent_disciplines(self):  # type: (...) -> str
        """The config_dependent_disciplines list which represents the list of disciplines that need to be configured after you configure"""
        return self.__config_dependent_disciplines

    @status.setter
    def status(self, status):
        """Setter of status"""
        self._update_status_dm(status)

    @property
    def all_input_structuring(self):
        """Property that is used to turn non-structuring variables into structuring when the flag is True."""
        return self.__all_input_structuring

    @all_input_structuring.setter
    def all_input_structuring(self, all_inp_struct: bool):
        """Setter of the all_input_structuring flag including a save of non-structuring variables values."""
        if self.__all_input_structuring is all_inp_struct:
            pass
        elif all_inp_struct is True:
            self._non_structuring_variables = {}
            self._set_structuring_variables_values(
                variables_dict=self._non_structuring_variables,
                variables_keys=self._get_non_structuring_variables_keys(),
                clear_variables_dict=True,
            )
            self.__all_input_structuring = True
        elif all_inp_struct is False:
            self._non_structuring_variables = None
            self.__all_input_structuring = False
        else:
            msg = 'all_input_structuring should be a boolean'
            raise ValueError(msg)

    def prepare_execution(self):
        """GEMSEO objects instanciation"""
        if self.discipline_wrapp is not None:
            if self.discipline_wrapp.discipline is not None:
                self.stored_cache = self.discipline_wrapp.discipline.cache
            # init gemseo discipline if it has not been created yet
            cache_type = self.get_sosdisc_inputs(self.CACHE_TYPE)

            if not cache_type or cache_type.lower() == "none":  # required for compatibility with old studies
                cache_type = Discipline.CacheType.NONE
            self.discipline_wrapp.create_gemseo_discipline(
                proxy=self,
                reduced_dm=self.ee.dm.reduced_dm,
                cache_type=cache_type,
                cache_file_path=self.get_sosdisc_inputs(self.CACHE_FILE_PATH),
            )
            self.add_status_observers_to_gemseo_disc()

        # else:
        #     # TODO : this should only be necessary when changes in structuring
        #     # variables happened?
        #     self.set_wrapper_attributes(self.discipline_wrapp.wrapper)
        #
        if self._reset_cache:
            # set new cache when cache_type have changed (self._reset_cache
            # == True)
            self.set_cache(self.discipline_wrapp.discipline, self.get_sosdisc_inputs(self.CACHE_TYPE))
            if self.get_sosdisc_inputs(self.CACHE_TYPE) == Discipline.CacheType.NONE and self.dm.cache_map is not None:
                self.delete_cache_in_cache_map()
        else:
            if self.stored_cache is not None:
                self.discipline_wrapp.discipline.cache = self.stored_cache
        if self._reset_linearization_mode:
            self.discipline_wrapp.discipline.linearization_mode = self.get_sosdisc_inputs(self.LINEARIZATION_MODE)
        #             if self._reset_debug_mode:
        #                 # update default values when changing debug modes between executions
        #                 to_update_debug_mode = self.get_sosdisc_inputs(self.DEBUG_MODE, in_dict=True, full_name=True)
        #                 self.discipline_wrapp.update_default_from_dict(to_update_debug_mode)
        #     # set the status to pending on GEMSEO side (so that it does not
        #     # stay on DONE from last execution)
        #     self.discipline_wrapp.discipline.status = Discipline.ExecutionStatus.Status.PENDING

        # clear the proxy from the wrapper before execution
        self.clear_proxy_from_wrapper()
        # status and flags
        self.status = self.discipline_wrapp.discipline.execution_status.value
        self._reset_cache = False
        self._reset_debug_mode = False
        self._reset_linearization_mode = False

        self.set_residuals_variables()

    def set_residuals_variables(self):
        """

        Set the residuals variables to the MDO Discipline
        residual_variables and run_solve_residuals boolean

        """
        self.discipline_wrapp.discipline.residual_variables = self.get_sosdisc_inputs(self.RESIDUAL_VARIABLES).copy()
        self.discipline_wrapp.discipline.run_solves_residuals = self.get_sosdisc_inputs(self.RUN_SOLVE_RESIDUALS)

    def add_status_observers_to_gemseo_disc(self):
        """Add all observers that have been addes when gemseo discipline was not instanciated"""
        for observer in self.status_observers:
            if self.discipline_wrapp is not None and self.discipline_wrapp.discipline is not None:
                self.discipline_wrapp.discipline.execution_status.add_observer(observer)

    def set_cache(self, disc: Discipline, cache_type: str) -> None:
        """Instanciate and set cache for disc.

        Arguments:
            disc (Discipline): GEMSEO object to set cache
            cache_type (string): type of cache
        """
        cache_type = (
            Discipline.CacheType.NONE if cache_type.lower() == "none" else cache_type
        )  # required for compatibility with old studies
        if cache_type == Discipline.CacheType.HDF5:
            msg = "If the cache type is set to HDF5Cache, the cache_file path must be set"
            raise ValueError(msg)
        disc.set_cache(cache_type=cache_type)
        if cache_type == Discipline.CacheType.SIMPLE:
            disc.cache.compare_dict_of_arrays = dict_are_equal

    def delete_cache_in_cache_map(self):
        """If a cache has been written"""
        hashed_uid = self.get_cache_map_hashed_uid(self)
        self.dm.delete_hashed_id_in_cache_map(hashed_uid)

    def get_shared_namespace_list(self, data_dict):
        """
        Get the list of namespaces defined in the data_in or data_out when the visibility of the variable is shared

        Arguments:
            data_dict (Dict[dict]): data_in or data_out
        """
        shared_namespace_list = []

        for item in data_dict.values():
            self.__append_item_namespace(item, shared_namespace_list)

        return list(set(shared_namespace_list))

    def __append_item_namespace(self, item, ns_list):
        """
        Append the namespace if the visibility is shared

        Arguments:
            item (dict): element to append to the ns_list
            ns_list (List[Namespace]): list of namespaces [???]
        """
        if self.VISIBILITY in item and item[self.VISIBILITY] == self.SHARED_VISIBILITY:
            with contextlib.suppress(Exception):
                ns_list.append(item[self.NAMESPACE])

    def get_input_data_names(self, as_namespaced_tuple: bool = False, numerical_inputs=True) -> list[str]:
        """
        Returns:
            (List[string]) of input data full names based on i/o and namespaces declarations in the user wrapper
        """
        data_in = self.get_data_io_with_full_name(self.IO_TYPE_IN, as_namespaced_tuple)
        if numerical_inputs:
            return list(data_in.keys())
        return [
            key
            for key, value in data_in.items()
            if (not value[self.NUMERICAL] or (value[self.NUMERICAL] and value[self.RUN_NEEDED]))
        ]

    def get_input_data_names_and_defaults(self, as_namespaced_tuple: bool = False, numerical_inputs=True) -> dict[str:Any]:
        """

        Args:
            as_namespaced_tuple: bool to choose if we wqnt to keep tuple in keys to deal with multiple variable in inputs (gather case)
            numerical_inputs: If numerical inputs is False then we do not tajke numerical variables that are not needed during the run (filter all unnecessary variables for gemseo)

        Returns: dict of input variables and value

        """
        data_in = self.get_data_io_with_full_name(self.IO_TYPE_IN, as_namespaced_tuple)
        if numerical_inputs:
            return {key: value[self.DEFAULT] for key, value in data_in.items()}
        return {key: value[self.DEFAULT] for key, value in data_in.items() if not self.variable_is_numerical(value)}

    def variable_is_numerical(self, definition_input_dict):
        """

        Args:
            definition_input_dict : dict with all parameters to define the variable

        Returns: True if the variable is numerical or  not needed for the run

        """
        return definition_input_dict[self.NUMERICAL] and not definition_input_dict[self.RUN_NEEDED]

    def get_run_needed_input(self, as_namespaced_tuple: bool = False):
        data_in = self.get_data_io_with_full_name(self.IO_TYPE_IN, as_namespaced_tuple)

        return {
            key: value[self.DEFAULT]
            for key, value in data_in.items()
            if value[self.NUMERICAL] and value[self.RUN_NEEDED]
        }

    def get_output_data_names(self, as_namespaced_tuple: bool = False, numerical_inputs=True) -> list[str]:
        """
        Returns:
            (List[string]) outpput data full names based on i/o and namespaces declarations in the user wrapper
        """
        data_out = self.get_data_io_with_full_name(self.IO_TYPE_OUT, as_namespaced_tuple)

        if numerical_inputs:
            return list(data_out.keys())
        return [key for key, value in data_out.items() if not value[self.NUMERICAL]]

    def get_data_io_dict(self, io_type: str) -> dict:
        """
        Get the DESC_IN+NUM_DESC_IN+inst_desc_in or the DESC_OUT+inst_desc_out depending on the io_type

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT

        Returns:
            (Dict(dict)) data_in or data_out
        Raises:
            Exception if io_type
        """
        if io_type == self.IO_TYPE_IN:
            return self.get_data_in()
        if io_type == self.IO_TYPE_OUT:
            return self.get_data_out()
        msg = f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]'
        raise ProxyDisciplineException(msg)

    def get_data_io_dict_keys(self, io_type):
        """
        Get the DESC_IN+NUM_DESC_IN+inst_desc_in or the DESC_OUT+inst_desc_out keys depending on the io_type

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT

        Returns:
            (list) data_in or data_out keys
        """
        if io_type == self.IO_TYPE_IN:
            return self.get_data_in().keys()
        if io_type == self.IO_TYPE_OUT:
            return self.get_data_out().keys()
        msg = f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]'
        raise ProxyDisciplineException(msg)

    def get_data_io_dict_tuple_keys(self, io_type):
        """
        Get the DESC_IN+NUM_DESC_IN+inst_desc_in or the DESC_OUT+inst_desc_out tuple keys depending on the io_type

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT

        Returns:
            (list of keys) _data_in or _data_out tuple keys
        """
        if io_type == self.IO_TYPE_IN:
            return self._data_in.keys()
        if io_type == self.IO_TYPE_OUT:
            return self._data_out.keys()
        msg = f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]'
        raise ProxyDisciplineException(msg)

    def get_data_io_from_key(self, io_type, var_name):
        """
        Return the namespace and the data_in/data_out of a single variable (short name)

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
            var_name (string): short name of the variable
        Returns:
            (dict) data_in or data_out of the variable
        """
        data_io = self.get_data_io_dict(io_type)

        if var_name in data_io:
            return data_io[var_name]
        msg = f'No key matching with variable name {var_name} in the data_{io_type}'
        raise ProxyDisciplineException(msg)

    def get_variable_name_from_ns_key(self, io_type, ns_key):
        """ """
        return self.get_data_io_dict(io_type)[ns_key][self.VAR_NAME]

    def reload_io(self):
        """
        Create the data_in and data_out of the discipline with the DESC_IN/DESC_OUT, inst_desc_in/inst_desc_out
        and initialize GEMS grammar with it (with a filter for specific variable types)
        """
        # get new variables from inst_desc_in (dynamic variables)
        new_inputs = self.get_new_variables_in_dict(self.inst_desc_in, self.IO_TYPE_IN)

        # get variables from desc_in if it is the first configure (data_in is empty) : static variables
        if self._data_in == {}:
            desc_in = self.get_desc_in_out(self.IO_TYPE_IN)
            new_inputs.update(desc_in)
            # add numerical variables to the new inputs dict
            num_data_in = deepcopy(self.NUM_DESC_IN)
            new_inputs.update(num_data_in)

        if len(new_inputs) > 0:
            self.update_data_io_and_nsmap(new_inputs, self.IO_TYPE_IN)

        # get new variables from inst_desc_out (dynamic variables)
        new_outputs = self.get_new_variables_in_dict(self.inst_desc_out, self.IO_TYPE_OUT)
        # get variables from desc_out if it is the first configure (data_out is empty) : static variables
        if self._data_out == {}:
            desc_out = self.get_desc_in_out(self.IO_TYPE_OUT)

            new_outputs.update(desc_out)

        # add new outputs from inst_desc_out to data_out
        if len(new_outputs) > 0:
            self.update_data_io_and_nsmap(new_outputs, self.IO_TYPE_OUT)

    def update_dm_with_data_dict(self, data_dict):
        """
        Update data manager for this discipline with data_dict.

        Arguments:
            data_dict (Dict[dict]): item to update data manager with
        """
        self.dm.update_with_discipline_dict(self.disc_id, data_dict)

    def get_desc_in_out(self, io_type):
        """
        Retrieves information from ProxyDiscipline and/or wrapper DESC_IN to fill data_in

        Argument:
            io_type : 'string' . indicates whether we are interested in desc_in or desc_out
        """
        if io_type == self.IO_TYPE_IN:
            _desc = deepcopy(self.DESC_IN) if self.DESC_IN else {}
            if self.discipline_wrapp and self.discipline_wrapp.wrapper and self.discipline_wrapp.wrapper.DESC_IN:
                _desc.update(deepcopy(self.discipline_wrapp.wrapper.DESC_IN))
            return _desc
        if io_type == self.IO_TYPE_OUT:
            _desc = deepcopy(self.DESC_OUT) if self.DESC_OUT else {}
            if self.discipline_wrapp and self.discipline_wrapp.wrapper and self.discipline_wrapp.wrapper.DESC_OUT:
                _desc.update(deepcopy(self.discipline_wrapp.wrapper.DESC_OUT))
            return _desc
        msg = f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]'
        raise ProxyDisciplineException(msg)

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
        return [(key, id(v[self.NS_REFERENCE])) for key, v in short_name_data_dict.items()]

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
            msg = f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]'
            raise ProxyDisciplineException(msg)

    def _restart_data_io_to_disc_io(self, io_type=None):
        """
        Restarts the _data_in/_data_out to contain only the variables referenced in short names in _io_ns_map_in/_io_ns_map_out.

        Arguments:
            io_type (string) : IO_TYPE_IN or IO_TYPE_OUT

        Raises:
            Exception if io_type is not IO_TYPE_IN or IO_TYPE_OUT
        """
        if io_type is None:
            io_types = [self.IO_TYPE_IN, self.IO_TYPE_OUT]
        elif io_type == self.IO_TYPE_IN or io_type == self.IO_TYPE_OUT:
            io_types = [io_type]
        else:
            msg = f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]'
            raise ProxyDisciplineException(msg)
        if self.IO_TYPE_IN in io_types:
            self._data_in = {(key, id_ns): self._data_in[key, id_ns] for key, id_ns in self._io_ns_map_in.items()}

        if self.IO_TYPE_OUT in io_types:
            self._data_out = {(key, id_ns): self._data_out[key, id_ns] for key, id_ns in self._io_ns_map_out.items()}

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
            msg = f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]'
            raise ProxyDisciplineException(msg)

        if data_dict_in_short_names:
            error_msg = 'data_dict_in_short_names for uodate_data_io not implemented'
            self.logger.error(error_msg)
            raise ProxyDisciplineException(error_msg)
        data_io.update(data_dict)

    def get_new_variables_in_dict(self, var_dict, io_type):
        """

        Args:
            var_dict: The variable dict to compare with data_io
            io_type: the type of variable input or output

        Returns:
            the dict filtered with variables that are not yet in data_io
        """
        return {
            key: value
            for key, value in var_dict.items()
            if key not in self.get_data_io_dict_keys(io_type) and key not in self.get_data_io_dict_tuple_keys(io_type)
        }

    def update_data_io_and_nsmap(self, new_data_dict, io_type):
        """Function to update data_in with a new data_dict and update also the ns_map related to the tuple key in the data_in"""
        self.set_shared_namespaces_dependencies(new_data_dict)
        completed_new_inputs = self._prepare_data_dict(io_type, new_data_dict)

        var_ns_tuples = self._extract_var_ns_tuples(completed_new_inputs)
        self._update_io_ns_map(var_ns_tuples, io_type)
        self.update_dm_with_data_dict(completed_new_inputs)
        self._update_data_io(zip(var_ns_tuples, completed_new_inputs.values()), io_type)

        self.build_simple_data_io(io_type)

    def get_built_disciplines_ids(self):
        """Returns: (List[string]) the names of the sub proxies."""
        return [disc.name for disc in self.proxy_disciplines]

    def get_proxy_disciplines(self):
        """Returns: (List[ProxyDiscipline]) the list of children sub proxies"""
        return self.proxy_disciplines

    def get_sub_proxy_disciplines(self, disc_list=None):
        """
        Recursively returns all descendancy of sub proxies

        Arguments:
            disc_list (List[ProxyDiscipline]): current list of descendancy of sub proxies

        Returns:
            (List[ProxyDiscipline]): complete descendancy of sub proxies
        """
        if disc_list is None:
            disc_list = []
        for disc in self.proxy_disciplines:
            disc_list.append(disc)
            disc.get_sub_proxy_disciplines(disc_list)
        return disc_list

    @property
    def ordered_disc_list(self):
        """
        Property to obtain the ordered list of disciplines by default, for a ProxyDiscipline it is the order of
        sub proxy disciplines
        """
        return self.proxy_disciplines

    def add_discipline(self, disc):
        """
        Add a discipline to the children sub proxies and set self as father executor.

        Arguments:
            disc (ProxyDiscipline): discipline to add
        """
        self.proxy_disciplines.append(disc)
        disc.set_father_executor(self)
        # self._check_if_duplicated_disc_names()

    def add_discipline_list(self, disc_list):
        """
        Add a list of disciplines to the children sub proxies and set self as father executor.

        Arguments:
            disc_list (List[ProxyDiscipline]): disciplines to add
        """
        for disc in disc_list:
            self.add_discipline(disc)

    def set_shared_namespaces_dependencies(self, data_dict):
        """
        Set dependencies of shared inputs and outputs in ns_manager

        Arguments:
            data_dict (Dict[dict]): data_in or data_out
        """
        shared_namespace_list = self.get_shared_namespace_list(data_dict)
        self.ee.ns_manager.add_dependencies_to_shared_namespace(self, shared_namespace_list)

    def add_variables(self, data_dict, io_type, clean_variables=True):
        """
        Add dynamic inputs/outputs in ins_desc_in/ints_desc_out and remove old dynamic inputs/outputs

        Arguments:
            data_dict (Dict[dict]): new dynamic inputs/outputs
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
            clean_variables (bool): flag to remove old variables from data_in/data_out, inst_desc_in/inst_desc_out, datamanger
        """
        variables_to_remove = []
        if io_type == self.IO_TYPE_IN:
            variables_to_remove = [key for key in self.inst_desc_in if key not in data_dict]
            self.inst_desc_in.update(data_dict)
        elif io_type == self.IO_TYPE_OUT:
            variables_to_remove = [key for key in self.inst_desc_out if key not in data_dict]
            self.inst_desc_out.update(data_dict)

        if clean_variables:
            self.clean_variables(variables_to_remove, io_type)

    def add_inputs(self, data_dict, clean_inputs=True):
        """
        Add dynamic inputs

        Arguments:
            data_dict (Dict[dict]): new dynamic inputs
            clean_variables (bool): flag to remove old variables from data_in, inst_desc_in and datamanger
        """
        self.add_variables(data_dict, self.IO_TYPE_IN, clean_variables=clean_inputs)

    def add_outputs(self, data_dict, clean_outputs=True):
        """
        Add dynamic outputs

        Arguments:
            data_dict (Dict[dict]): new dynamic outputs
            clean_variables (bool): flag to remove old variables from data_out, inst_desc_out and datamanger
        """
        self.add_variables(data_dict, self.IO_TYPE_OUT, clean_variables=clean_outputs)

    def clean_variables(self, var_name_list: list[str], io_type: str):
        """
        Remove variables from data_in/data_out, inst_desc_in/inst_desc_out and datamanger

        Arguments:
            var_name_list (List[string]): variable names to clean
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
        """
        for var_name in var_name_list:
            if io_type == self.IO_TYPE_IN:
                del self.inst_desc_in[var_name]
                self.ee.dm.remove_keys(self.disc_id, self.get_var_full_name(var_name, self.get_data_in()), io_type)

                del self._data_in[var_name, self._io_ns_map_in[var_name]]
                del self._io_ns_map_in[var_name]

            elif io_type == self.IO_TYPE_OUT:
                if var_name in self.inst_desc_out:
                    del self.inst_desc_out[var_name]
                self.ee.dm.remove_keys(self.disc_id, self.get_var_full_name(var_name, self.get_data_out()), io_type)

                del self._data_out[var_name, self._io_ns_map_out[var_name]]
                del self._io_ns_map_out[var_name]

            if var_name in self._structuring_variables:
                del self._structuring_variables[var_name]

        self.build_simple_data_io(io_type)

    def update_default_value(self, var_name: str, io_type: str, new_default_value):
        """
        Update DEFAULT and VALUE of var_name in data_io

        Arguments:
            var_name (string): variable names to clean
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
            new_default_value: value to update VALUE and DEFAULT with
        """
        if var_name in self.get_data_io_dict(io_type):
            self.get_data_io_dict(io_type)[var_name][self.DEFAULT] = new_default_value
            self.get_data_io_dict(io_type)[var_name][self.VALUE] = new_default_value

    def assign_proxy_to_wrapper(self):
        """Assign the proxy (self) to the SoSWrapp for configuration actions."""
        if self.discipline_wrapp is not None and self.discipline_wrapp.wrapper is not None:
            self.discipline_wrapp.wrapper.assign_proxy(self)

    def clear_proxy_from_wrapper(self):
        """Clears the proxy (self) from the SoSWrapp object for serialization and execution."""
        if self.discipline_wrapp is not None and self.discipline_wrapp.wrapper is not None:
            self.discipline_wrapp.wrapper.clear_proxy()

    # -- Configure handling
    def configure(self):
        """Configure the ProxyDiscipline"""
        self.assign_proxy_to_wrapper()
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
            for disc in self.config_dependent_disciplines:
                disc.set_configure_status(False)

    def __check_all_data_integrity(self):
        """
        generic data integrity_check where we call different generic function to check integrity
        + specific data integrity by discipline
        """
        data_integrity = self.__generic_check_data_integrity()
        # Test specific data integrity only if generic data integrity is OK
        # This will prevent to launch specific data integrity if all values are None
        # And to reimplement some data integrity already present in the generic method
        # example : the dataframe has not the right column compared to the df_descriptor (generic data integrity will raise an error)
        # the specific data integrity check values of this column but should verify before if the column exists
        if data_integrity:
            self.check_data_integrity()

    def check_data_integrity(self):
        if self.discipline_wrapp is not None:
            self.discipline_wrapp.check_data_integrity()

    def __generic_check_data_integrity(self):
        """Generic check data integrity of the variables that you own ( the model origin of the variable is you)"""
        data_integrity = True
        data_in_full_name = self.get_data_io_with_full_name(self.IO_TYPE_IN)
        for var_fullname in data_in_full_name:
            var_data_dict = self.dm.get_data(var_fullname)

            if var_data_dict['model_origin'] == self.disc_id:
                #                 check_integrity_msg = check_data_integrity_cls.check_variable_type_and_unit(var_data_dict)
                check_integrity_msg = self.check_data_integrity_cls.check_variable_value(
                    var_data_dict, self.ee.check_data_integrity
                )
                if check_integrity_msg:
                    data_integrity = False
                self.dm.set_data(var_fullname, self.CHECK_INTEGRITY_MSG, check_integrity_msg)
        return data_integrity

    def set_numerical_parameters(self):
        """Set numerical parameters of the ProxyDiscipline defined in the NUM_DESC_IN"""
        if self._data_in != {}:
            self.linearization_mode = self.get_sosdisc_inputs(self.LINEARIZATION_MODE)

            self.update_reset_cache()

            self.update_reset_debug_mode()

            self.update_reset_linearization_mode()

    def update_reset_linearization_mode(self) -> None:
        """Update the reset_linearization_mode boolean if linearization mode has changed"""
        linearization_mode = self.get_sosdisc_inputs(self.LINEARIZATION_MODE)
        stucturing_variable_linearization_mode = self._structuring_variables[self.LINEARIZATION_MODE]
        if linearization_mode != stucturing_variable_linearization_mode and not (
            linearization_mode == "auto" and self._structuring_variables[self.LINEARIZATION_MODE] is None
        ):
            self._reset_linearization_mode = True
        self.logger.debug(f"Discipline {self.sos_name} set to linearization mode {linearization_mode}")

    def update_reset_debug_mode(self):
        """Update the reset_debug_mode boolean if debug mode has changed + logger"""
        # Debug mode logging and recursive setting (priority to the parent)
        debug_mode = self.get_sosdisc_inputs(self.DEBUG_MODE)
        if (debug_mode or self._structuring_variables[self.DEBUG_MODE]) and (
            debug_mode != self._structuring_variables[self.DEBUG_MODE]
        ):  # not necessary on first config
            self._reset_debug_mode = True
            # logging
            if debug_mode:
                if debug_mode == "all":
                    for mode in self.AVAILABLE_DEBUG_MODE:
                        if mode not in ["", "all"]:
                            self.logger.info(f'Discipline {self.sos_name} set to debug mode {mode}')
                else:
                    self.logger.info(f'Discipline {self.sos_name} set to debug mode {debug_mode}')

    def update_reset_cache(self):
        """Update the reset_cache boolean if cache type has changed"""
        cache_type = self.get_sosdisc_inputs(self.CACHE_TYPE)
        if cache_type != self._structuring_variables[self.CACHE_TYPE]:
            self._reset_cache = True
            self._set_children_cache = True

    def set_debug_mode_rec(self, debug_mode: str):
        """Set debug mode recursively to children with priority to parent"""
        for disc in self.proxy_disciplines:
            disc_in = disc.get_data_in()
            if ProxyDiscipline.DEBUG_MODE in disc_in:
                self.dm.set_data(
                    self.get_var_full_name(self.DEBUG_MODE, disc_in), self.VALUE, debug_mode, check_value=False
                )
                disc.set_debug_mode_rec(debug_mode=debug_mode)

    def set_linearization_mode_rec(self, linearization_mode: str):
        """Set linearization mode recursively to children with priority to parent + log"""
        for disc in self.proxy_disciplines:
            disc_in = disc.get_data_in()
            if self.LINEARIZATION_MODE in disc_in:
                self.dm.set_data(
                    self.get_var_full_name(self.LINEARIZATION_MODE, disc_in),
                    self.VALUE,
                    linearization_mode,
                    check_value=False,
                )
                disc.set_linearization_mode_rec(linearization_mode=linearization_mode)

    def setup_sos_disciplines(self):
        """
        Method to be overloaded to add dynamic inputs/outputs using add_inputs/add_outputs methods.
        If the value of an input X determines dynamic inputs/outputs generation, then the input X is structuring and the item 'structuring':True is needed in the DESC_IN
        DESC_IN = {'X': {'structuring':True}}
        """
        self.discipline_wrapp.setup_sos_disciplines()

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
                    f'Try to set a default value for the variable {short_key} in {self.sos_name} which is not an input of this discipline '
                )

    # -- data handling section
    def reset_data(self):
        """Reset instance data attributes of the discipline to empty dicts."""
        self.inst_desc_in = {}
        self.inst_desc_out = {}
        self._data_in = {}
        self._simple_data_in = {}
        self._data_out = {}
        self._simple_data_out = {}
        self._io_ns_map_in = {}
        self._io_ns_map_out = {}

        self._structuring_variables = {}

    def get_data_in(self):
        """ "
        _simple_data_in getter
        """
        return self._simple_data_in

    def get_io_ns_map(self, io_type):
        """

        Args:
            io_type: in or out

        Returns: the _io_ns_map_in or _io_ns_map_out depending on the io_type

        """
        if io_type == self.IO_TYPE_IN:
            return self._io_ns_map_in
        if io_type == self.IO_TYPE_OUT:
            return self._io_ns_map_out
        return None

    def build_simple_data_io(self, io_type):
        """

        Args:
            io_type: in or out string

        Returns: Buiold the simple_data_in dict which is the data_in withotu the tuple as key but only the name of the variable

        """
        data_io = self.get_data_io_with_full_name(io_type, True)
        io_ns_map = self.get_io_ns_map(io_type)

        if io_type == self.IO_TYPE_IN:
            self._simple_data_in = {var_name: data_io[var_name, id_ns] for (var_name, id_ns) in io_ns_map.items()}
        elif io_type == self.IO_TYPE_OUT:
            self._simple_data_out = {var_name: data_io[var_name, id_ns] for (var_name, id_ns) in io_ns_map.items()}

    def get_data_out(self):
        """
        _data_out getter
        #TODO: RENAME THIS METHOD OR ADD MODES 's'/'f'/'t' (short/full/tuple) as only the discipline dict and not subprocess is output
        """
        return self._simple_data_out

    def get_data_io_with_full_name(self, io_type, as_namespaced_tuple=False):
        """
        returns a version of the data_in/data_out of discipline with variable full names

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT

        Return:
            data_io_full_name (Dict[dict]): data_in/data_out with variable full names
        """
        if io_type == self.IO_TYPE_IN:
            if as_namespaced_tuple:
                return self._data_in
            return self.ns_tuples_to_full_name_keys(self._data_in)
        if io_type == self.IO_TYPE_OUT:
            if as_namespaced_tuple:
                return self._data_out
            return self.ns_tuples_to_full_name_keys(self._data_out)
        msg = 'Unknown io type'
        raise ValueError(msg)

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
        return data_io_full_name[full_name][data_name]

    def initialize_gemseo_io(self) -> None:
        """Create the GEMSEO IO object and fills the grammars.

        This method must be call before creating the coupling structure.
        """
        self.io = IO(
            discipline_class=None,
            discipline_name=self.name,
            grammar_type=self.SOS_GRAMMAR_TYPE,
        )
        self.io.input_grammar.update_from_names(self.get_input_data_names())
        self.io.output_grammar.update_from_names(self.get_output_data_names())

    def get_ns_reference(self, visibility, namespace=None):
        """
        Get namespace reference by consulting the namespace_manager

        Arguments:
            visibility (string): visibility to get local or shared namespace
            namespace (Namespace): namespace in case of shared visibility
        """
        ns_manager = self.ee.ns_manager

        if visibility == self.LOCAL_VISIBILITY or visibility == self.INTERNAL_VISIBILITY:
            return ns_manager.get_local_namespace(self)

        if visibility == self.SHARED_VISIBILITY:
            return ns_manager.get_shared_namespace(self, namespace)
        return None

    def apply_visibility_ns(self, io_type):
        """
        Consult the namespace_manager to apply the namespace depending on the variable visibility

        Arguments:
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
        """
        dict_in_keys = self.get_data_io_dict_keys(io_type)
        ns_manager = self.ee.ns_manager

        dict_out_keys = []
        for key in dict_in_keys:
            namespaced_key = ns_manager.get_namespaced_variable(self, key, io_type)
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
        new_data_dict = {}
        for key, curr_data in data_dict.items():
            # Kill the potential object link here , better way to do that ?
            new_data = dict(curr_data.items())
            data_keys = new_data.keys()
            new_data[self.IO_TYPE] = io_type
            new_data[self.TYPE_METADATA] = None
            if isinstance(key, tuple):
                new_data[self.VAR_NAME] = key[0]
            else:
                new_data[self.VAR_NAME] = key
            if self.USER_LEVEL not in data_keys:
                new_data[self.USER_LEVEL] = 1
            if self.RANGE not in data_keys:
                new_data[self.RANGE] = None
            if self.UNIT not in data_keys:
                new_data[self.UNIT] = None
            if self.DESCRIPTION not in data_keys:
                new_data[self.DESCRIPTION] = None
            if self.POSSIBLE_VALUES not in data_keys:
                new_data[self.POSSIBLE_VALUES] = None
            if new_data[self.TYPE] in ['array', 'dict', 'dataframe']:
                if self.DATAFRAME_DESCRIPTOR not in data_keys:
                    new_data[self.DATAFRAME_DESCRIPTOR] = None
                if self.DATAFRAME_EDITION_LOCKED not in data_keys:
                    new_data[self.DATAFRAME_EDITION_LOCKED] = True
            # For dataframes but also dict of dataframes...
            if new_data[self.TYPE] in ['dict', 'dataframe'] and self.DF_EXCLUDED_COLUMNS not in data_keys:
                new_data[self.DF_EXCLUDED_COLUMNS] = self.DEFAULT_EXCLUDED_COLUMNS

            if self.DISCIPLINES_FULL_PATH_LIST not in data_keys:
                new_data[self.DISCIPLINES_FULL_PATH_LIST] = []
            if self.VISIBILITY not in data_keys:
                new_data[self.VISIBILITY] = self.LOCAL_VISIBILITY
            if self.DEFAULT not in data_keys:
                if new_data[self.VISIBILITY] == self.INTERNAL_VISIBILITY:
                    msg = f'The variable {key} in discipline {self.sos_name} must have a default value because its visibility is Internal'
                    raise ValueError(msg)
                new_data[self.DEFAULT] = None
            else:
                new_data[self.VALUE] = new_data[self.DEFAULT]
            # -- Initialize VALUE to None by default
            if self.VALUE not in data_keys:
                new_data[self.VALUE] = None
            if self.COUPLING not in data_keys:
                new_data[self.COUPLING] = False
            if self.OPTIONAL not in data_keys:
                new_data[self.OPTIONAL] = False
            if self.NUMERICAL not in data_keys:
                new_data[self.NUMERICAL] = False
            if new_data[self.NUMERICAL] and self.RUN_NEEDED not in data_keys:
                new_data[self.RUN_NEEDED] = False
            if self.META_INPUT not in data_keys:
                new_data[self.META_INPUT] = False

            # -- Outputs are not EDITABLE
            if self.EDITABLE not in data_keys:
                if new_data[self.VISIBILITY] == self.INTERNAL_VISIBILITY:
                    new_data[self.EDITABLE] = False
                else:
                    new_data[self.EDITABLE] = io_type == self.IO_TYPE_IN
            # -- Add NS_REFERENCE
            if new_data[self.VISIBILITY] not in self.AVAILABLE_VISIBILITIES:
                var_name = new_data[self.VAR_NAME]
                visibility = new_data[self.VISIBILITY]
                raise ValueError(
                    self.sos_name
                    + '.'
                    + var_name
                    + ': '
                    + self.VISIBILITY
                    + str(visibility)
                    + ' not in allowed visibilities: '
                    + str(self.AVAILABLE_VISIBILITIES)
                )
            if self.NAMESPACE in data_keys:
                new_data[self.NS_REFERENCE] = self.get_ns_reference(new_data[self.VISIBILITY], new_data[self.NAMESPACE])
            else:
                new_data[self.NS_REFERENCE] = self.get_ns_reference(new_data[self.VISIBILITY])

            # store structuring variables in self._structuring_variables
            if self.STRUCTURING in data_keys and new_data[self.STRUCTURING] is True:
                if new_data[self.IO_TYPE] == self.IO_TYPE_IN:
                    self._structuring_variables[key] = None
                del new_data[self.STRUCTURING]
            if self.CHECK_INTEGRITY_MSG not in data_keys:
                new_data[self.CHECK_INTEGRITY_MSG] = ''

            # initialize formula
            if self.FORMULA not in data_keys:
                new_data[self.FORMULA] = None
            if self.IS_FORMULA not in data_keys:
                new_data[self.IS_FORMULA] = False
            if self.IS_EVAL not in data_keys:
                new_data[self.IS_EVAL] = False
            new_data_dict[key] = new_data
        return new_data_dict

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
        # TODO: refactor
        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            if full_name_keys:
                keys = list(self.get_data_io_with_full_name(self.IO_TYPE_IN).keys())  # discipline and subprocess
            else:
                keys = list(self.get_data_in().keys())  # discipline only
            in_dict = True
        inputs = self._get_sosdisc_io(keys, io_type=self.IO_TYPE_IN, full_name_keys=full_name_keys)
        if in_dict:
            # return inputs in an dictionary
            return inputs
        # return inputs in an ordered tuple (default)
        if len(inputs) > 1:
            return list(inputs.values())
        return next(iter(inputs.values()))

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
        # TODO: refactor
        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            if full_name_keys:
                keys = list(self.get_data_io_with_full_name(self.IO_TYPE_OUT).keys())  # discipline and subprocess
            else:
                keys = list(self.get_data_out().keys())  # discipline only
            in_dict = True
        outputs = self._get_sosdisc_io(keys, io_type=self.IO_TYPE_OUT, full_name_keys=full_name_keys)
        if in_dict:
            # return outputs in an dictionary
            return outputs
        # return outputs in an ordered tuple (default)
        if len(outputs) > 1:
            return list(outputs.values())
        return next(iter(outputs.values()))

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
        # TODO: refactor
        if isinstance(keys, str):
            keys = [keys]

        query_keys = keys if full_name_keys else self._convert_list_of_keys_to_namespace_name(keys, io_type)

        values_dict = {}
        for key, q_key in zip(keys, query_keys):
            if q_key not in self.dm.data_id_map:
                msg = f'The key {q_key} for the discipline {self.get_disc_full_name()} is missing in the data manager'
                raise ValueError(msg)
            # get data in local_data during run or linearize steps
            # #TODO: this should not be possible in command line mode, is it possible in the GUI?
            if self.status in [self.STATUS_RUNNING, self.STATUS_LINEARIZE]:
                # a variable is in the local_data if the variable is not numerical, do not need of numerical variables in the run phase
                if not self.variable_is_numerical(self.dm.get_data(q_key)):
                    values_dict[key] = self.discipline_wrapp.discipline.io.data[q_key]
            # get data in data manager during configure step
            else:
                values_dict[key] = self.dm.get_value(q_key)
        return values_dict

    def _update_type_metadata(self):
        """Update metadata of values not supported by GEMS (for cases where the data has been converted by the coupling)"""
        disc_in = self.get_data_in()
        for var_name in disc_in:
            var_f_name = self.get_var_full_name(var_name, disc_in)
            var_type = self.dm.get_data(var_f_name, self.VAR_TYPE_ID)
            if var_type in self.NEW_VAR_TYPE:
                if self.dm.get_data(var_f_name, self.TYPE_METADATA) is not None:
                    disc_in[var_name][self.TYPE_METADATA] = self.dm.get_data(var_f_name, self.TYPE_METADATA)

        disc_out = self.get_data_out()
        for var_name in disc_out:
            var_f_name = self.get_var_full_name(var_name, disc_out)
            var_type = self.dm.get_data(var_f_name, self.VAR_TYPE_ID)
            if var_type in self.NEW_VAR_TYPE:
                if self.dm.get_data(var_f_name, self.TYPE_METADATA) is not None:
                    disc_out[var_name][self.TYPE_METADATA] = self.dm.get_data(var_f_name, self.TYPE_METADATA)

    def _update_study_ns_in_varname(self, names):
        """
        Updates the study name in the variable input names.

        Arguments:
            names (List[string]): names to update
        """
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

    def clean_dm_from_disc(self):
        """Clean ProxyDiscipline in datamanager's disciplines_dict and data_in/data_out keys"""
        self.dm.clean_from_disc(self.disc_id)

    def _set_dm_disc_info(self):
        """Set info of the ProxyDiscipline in datamanager"""
        disc_ns_name = self.get_disc_full_name()
        disc_dict_info = {}
        disc_dict_info[self.REFERENCE] = self
        disc_dict_info['classname'] = self.__class__.__name__
        disc_dict_info[self.MODEL_NAME_FULL_PATH] = self.get_module()
        disc_dict_info['disc_label'] = self.get_disc_label()
        disc_dict_info['treeview_order'] = 'no'
        disc_dict_info[self.NS_REFERENCE] = self.ee.ns_manager.get_local_namespace(self)
        self.disc_id = self.dm.update_disciplines_dict(self.disc_id, disc_dict_info, disc_ns_name)

    def _set_dm_cache_map(self):
        """Update cache_map dict in DM with cache and its children recursively"""
        discipline = self.discipline_wrapp.discipline if self.discipline_wrapp is not None else None
        if discipline is not None:
            self._store_cache_with_hashed_uid(discipline)
        # store children cache recursively
        for disc in self.proxy_disciplines:
            disc._set_dm_cache_map()

    def _store_cache_with_hashed_uid(self, disc):
        """Generate hashed uid and store cache in DM"""
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
        return self.dm.generate_hashed_uid(disc_info_list)

    def get_var_full_name(self, var_name, disc_dict):
        """Get namespaced variable from namespace and var_name in disc_dict"""
        ns_reference = disc_dict[var_name][self.NS_REFERENCE]
        complete_var_name = disc_dict[var_name][self.VAR_NAME]
        return self.ee.ns_manager.compose_ns([ns_reference.value, complete_var_name])

    def get_input_var_full_name(self, var_name):
        """Get namespaced input variable"""
        return self.get_var_full_name(var_name, self.get_data_in())

    def get_output_var_full_name(self, var_name):
        """Get namespaced input variable"""
        return self.get_var_full_name(var_name, self.get_data_out())

    def get_var_display_name(self, var_name, disc_dict):
        """Get namespaced variable from display namespace and var_name in disc_dict"""
        ns_reference = disc_dict[var_name][self.NS_REFERENCE]
        complete_var_name = disc_dict[var_name][self.VAR_NAME]
        return self.ee.ns_manager.compose_ns([ns_reference.get_display_value(), complete_var_name])

    def ns_tuples_to_full_name_keys(self, in_dict):
        """
        Converts the keys of the input dictionary from tuples (var_short_name, id(ns_ref)) to strings var_full_name.

        Arguments:
            in_dict (dict[Any]): the input dictionary whose keys are tuples (var_short_name, id(ns_ref))

        Returns:
            dict[Any]: the dictionary with same values and full name keys
        """
        return {
            self.ee.ns_manager.ns_tuple_to_full_name(var_ns_tuple): value for var_ns_tuple, value in in_dict.items()
        }

    def update_from_dm(self):
        """Update all disciplines with datamanager information"""
        self.__check_all_data_integrity()

        disc_in = self.get_data_in()
        for var_name in disc_in:
            var_f_name = self.get_var_full_name(var_name, disc_in)
            try:
                default_val = self.dm.data_dict[self.dm.get_data_id(var_f_name)][self.DEFAULT]
            except Exception:
                default_val = None
            if self.dm.get_value(var_f_name) is None and default_val is not None:
                disc_in[var_name][self.VALUE] = default_val
            else:
                # update from dm for all proxy_disciplines to load all data
                disc_in[var_name][self.VALUE] = self.dm.get_value(var_f_name)
        # -- update sub-disciplines
        for discipline in self.proxy_disciplines:
            discipline.update_from_dm()

    # -- Ids and namespace handling
    def get_disc_full_name(self):
        """Return: (string) the discipline name with full namespace"""
        return self.ee.ns_manager.get_local_namespace_value(self)

    def get_disc_display_name(self, exec_display=False):
        """Return: (string) the discipline name with either the display namespace or the exec namespace"""
        if exec_display:
            return self.ee.ns_manager.get_local_namespace_value(self)
        return self.ee.ns_manager.get_display_namespace_value(self)

    def get_disc_id_from_namespace(self):
        """Return: (string) the discipline id"""
        return self.ee.dm.get_discipline_ids_list(self.get_disc_full_name())

    def get_single_data_io_string_for_disc_uid(self, disc):
        """Return: (List[string]) of anonimated input and output keys for serialisation purpose"""
        if isinstance(disc, ProxyDiscipline):
            input_list_anonimated = [key.split(self.ee.study_name, 1)[-1] for key in disc.get_input_data_names()]
            output_list_anonimated = [key.split(self.ee.study_name, 1)[-1] for key in disc.get_output_data_names()]
        else:
            input_list_anonimated = [key.split(self.ee.study_name, 1)[-1] for key in disc.io.input_grammar.names]
            output_list_anonimated = [key.split(self.ee.study_name, 1)[-1] for key in disc.io.output_grammar.names]

        input_list_anonimated.sort()
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
        if isinstance(keys, str):
            variables = [self._convert_to_namespace_name(keys, io_type)]
        else:
            variables = [self._convert_to_namespace_name(key, io_type) for key in keys]
        return variables

    def _convert_to_namespace_name(self, key, io_type):
        """Convert to namespace with coupling_namespace management
        Using a key (variables name) and reference_data (yaml in or out),
        build the corresponding namespaced key using the visibility property

        Arguments:
            key (string): variable name

        Return:
            (string) the variable namespace name
        """
        # Refactor  variables keys with namespace
        return self.ee.ns_manager.get_namespaced_variable(self, key, io_type)

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
            # no more configure status in gemseo !!
            # if self.discipline_wrapp is not None and self.discipline_wrapp.discipline is not None:
            #     self.discipline_wrapp.discipline.execution_status.value = status

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

    def set_status_from_discipline(self):
        """Update status of self and children sub proxies by retreiving the status of the GEMSEO objects."""
        for proxy_discipline in self.proxy_disciplines:
            proxy_discipline.set_status_from_discipline()
        self.status = self.get_status_after_configure()

    def get_status_after_configure(self):
        if self.discipline_wrapp is not None and self.discipline_wrapp.discipline is not None:
            return self.discipline_wrapp.discipline.execution_status.value
        return self._status

    def add_status_observer(self, observer):
        """
        Observer has to be set before execution (and prepare_execution) and the discipline does not exist.
        We store observers in self.status_observers and add it to the mdodiscipline when it ies instanciated in prepare_execution
        """
        if self.discipline_wrapp is not None and self.discipline_wrapp.discipline is not None:
            self.discipline_wrapp.discipline.execution_status.add_observer(observer)

        if observer not in self.status_observers:
            self.status_observers.append(observer)

    def remove_status_observer(self, observer):
        """
        Remove the observer from the status_observers list
        And normally the mdodiscipline has already been instanciated and we can remove it.
        If not the case the mdodiscipline does not exist such as the observer
        """
        if observer in self.status_observers:
            self.status_observers.remove(observer)
        if self.discipline_wrapp is not None and self.discipline_wrapp.discipline is not None:
            self.discipline_wrapp.discipline.execution_status.remove_observer(observer)

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
            msg = f'Unkown maturity {maturity} for discipline {self.sos_name}'
            raise ProxyDisciplineException(msg)

    def get_maturity(self):
        """Get the maturity of the ProxyDiscipline (a discipline does not have any subdisciplines, only a coupling has)"""
        if hasattr(self, '_maturity'):
            return self._maturity
        if hasattr(self.discipline_wrapp, 'wrapper'):
            if hasattr(self.discipline_wrapp.wrapper, '_maturity'):
                return self.discipline_wrapp.wrapper._maturity
            return ''
        return ''

    def get_chart_filter_list(self):
        """
        Return a list of ChartFilter instance base on the inherited class post processing filtering capabilities

        Returns: List[ChartFilter]
        """
        if self.discipline_wrapp is not None and self.discipline_wrapp.wrapper is not None:
            self.assign_proxy_to_wrapper()  # to allow for direct calls after run, without reconfiguration
            return self.discipline_wrapp.wrapper.get_chart_filter_list()
        return []

    def get_post_processing_list(self, filters=None):
        """
        Return a list of post processing instance using the ChartFilter list given as parameter

        Arguments:
            filters: filters to apply during post processing making

        Returns:
            post processing instance list
        """
        if self.discipline_wrapp is not None and self.discipline_wrapp.wrapper is not None:
            self.assign_proxy_to_wrapper()  # to allow for direct calls after run, without reconfiguration
            return self.discipline_wrapp.wrapper.get_post_processing_list(filters)
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
        return ''

    def is_configured(self):
        """Return False if discipline needs to be configured, True if not"""
        is_proxy_configured = (
            self.get_configure_status()
            and not self.check_structuring_variables_changes()
            and self.check_configured_dependency_disciplines()
        )

        # condition of wrapper configuration allows to redefine is_configured method for simple discs at wrapper level
        if hasattr(self.discipline_wrapp, 'wrapper') and hasattr(self.discipline_wrapp.wrapper, 'is_configured'):
            is_wrapper_configured = self.discipline_wrapp.wrapper.is_configured()
        else:
            is_wrapper_configured = True

        return is_proxy_configured and is_wrapper_configured

    def check_configured_dependency_disciplines(self):
        """
        Check if config_dependency_disciplines are configured to know if i am configured
        Be careful using this capability to avoid endless loop of configuration
        """
        return all(disc.is_configured() for disc in self.config_dependency_disciplines)

    def add_disc_to_config_dependency_disciplines(self, disc):
        """
        Add a discipline to config_dependency_disciplines
        Be careful to endless configuraiton loop (small loops are checked but not with more than two disciplines)
        Do not add twice the same dsicipline
        """
        if disc == self:
            error_msg = f'Not possible to add self in the config_dependency_list for disc : {disc.get_disc_full_name()}'
            self.logger.error(error_msg)
            raise ProxyDisciplineException(error_msg)

        if self in disc.config_dependency_disciplines:
            error_msg = f'The discipline {disc.get_disc_full_name()} has already {self.get_disc_full_name()} in its config_dependency_list, it is not possible to add the discipline in config_dependency_list of myself'
            self.logger.error(error_msg)
            raise ProxyDisciplineException(error_msg)

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
        """Add a list to children_list"""
        for disc in disc_list:
            self.add_disc_to_config_dependency_disciplines(disc)

    def add_new_shared_ns(self, shared_ns):
        self.ee.ns_manager.add_new_shared_ns_for_disc(self, shared_ns)

    @staticmethod
    def _get_disciplines_to_configure(disc_list):
        """
        Get sub disciplines list to configure according to their is_configured method (coupling, eval, etc.) from a
        discipline list
        """
        return [disc for disc in disc_list if disc.configurator is None and not disc.is_configured()]

    def get_disciplines_to_configure(self):
        """Get sub disciplines list to configure according to their is_configured method (coupling, eval, etc.)"""
        return self._get_disciplines_to_configure(self.proxy_disciplines)

    def check_structuring_variables_changes(self):
        """
        Compare structuring variables stored in discipline with values in dm
        Return True if at least one structuring variable value has changed, False if not
        """
        _struct_var_changes = self._check_structuring_variables_changes(self._structuring_variables)
        if self.all_input_structuring:
            _struct_var_changes = _struct_var_changes or self._check_structuring_variables_changes(
                self._non_structuring_variables, variables_keys=self._get_non_structuring_variables_keys()
            )
        return _struct_var_changes

    def set_structuring_variables_values(self):
        """Store structuring variables values from dm in self._structuring_variables"""
        self._set_structuring_variables_values(self._structuring_variables)
        if self.all_input_structuring:
            self._set_structuring_variables_values(
                self._non_structuring_variables,
                variables_keys=self._get_non_structuring_variables_keys(),
                clear_variables_dict=True,
            )

    def _check_structuring_variables_changes(self, variables_dict, variables_keys=None):
        dict_values_dm = {
            key: self.get_sosdisc_inputs(key)
            for key in (variables_dict.keys() if variables_keys is None else variables_keys)
        }
        try:
            return dict_values_dm != variables_dict
        except ValueError:
            return not dict_are_equal(dict_values_dm, variables_dict)

    def _set_structuring_variables_values(self, variables_dict, variables_keys=None, clear_variables_dict=False):
        disc_in = self.get_data_in()
        keys_to_check = list(
            variables_dict.keys() if variables_keys is None else variables_keys
        )  # copy necessary in case dict is cleared
        if clear_variables_dict:
            variables_dict.clear()
        for struct_var in keys_to_check:
            if struct_var in disc_in:
                variables_dict[struct_var] = deepcopy(self.get_sosdisc_inputs(struct_var))

    def _get_non_structuring_variables_keys(self):
        """
        Method used to return the non-structuring variables of a discipline. Can be overloaded to add exceptions i.e.
        variables that should never be considered structuring even if the all_input_structuring flag is set to True.
        """
        return self.get_data_in().keys() - self._structuring_variables.keys()

    # ----------------------------------------------------
    # ----------------------------------------------------
    #  METHODS TO DEBUG DISCIPLINE
    # ----------------------------------------------------
    # ----------------------------------------------------

    #     def check_jacobian(self, input_data=None, derr_approx=Discipline.FINITE_DIFFERENCES,
    #                        step=1e-7, threshold=1e-8, linearization_mode='auto',
    #                        inputs=None, outputs=None, parallel=False,
    #                        n_processes=Discipline.N_CPUS,
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
        self.ee.ns_manager.remove_dependencies_after_disc_deletion(self, self.disc_id)
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
            'output_full_name_map': output_full_name_map,
        }
        wrapper.inst_desc_in = self.inst_desc_in
        wrapper.inst_desc_out = self.inst_desc_out

    def create_io_full_name_map(self):
        """
        Create an io_full_name_map as wel ass input_full_name_map and output_full_name_map for its sos_wrapp

        Return:
            input_full_name_map (Dict[Str]): dict whose keys are input short names and values are input full names
            output_full_name_map (Dict[Str]): dict whose keys are output short names and values are output full names
        """
        return {
            key: self.ee.ns_manager.ns_tuple_to_full_name((key, value)) for key, value in self._io_ns_map_in.items()
        }, {key: self.ee.ns_manager.ns_tuple_to_full_name((key, value)) for key, value in self._io_ns_map_out.items()}

    def get_module(self):
        """Obtain the module of the wrapper if it exists. useful for postprocessing factory and treenode"""
        if self.discipline_wrapp is not None and self.discipline_wrapp.wrapper is not None:
            disc_module = self.discipline_wrapp.wrapper.__module__

        else:
            # for discipline not associated to wrapper (proxycoupling for
            # example)
            disc_module = self.__module__
        # return the replace sostrades_core just for documentation (linked
        # ontology is the one from integration)
        #         return disc_module.replace(
        #             'sostrades_core', 'sos_trades_core')
        return disc_module

    # useful for debugging

    def get_shared_ns_dict(self):
        return self.ee.ns_manager.get_associated_ns(self)

    def get_disc_label(self):
        """Get the label of the discipline which will be displayed in the GUI"""
        return self.get_disc_full_name()

    def get_disc_full_path(self):
        """Get the discipline full path which is a combination of the module and the label of the discipline"""
        return f'{self.get_module()} : {self.get_disc_label()}'

    def display_proxy_subtree(self, callback=None):
        """
        Display in a treeview fashion the subtree of proxy_disciplines of the discipline, usually called from the
        root_process.
        Example: ee.root_process.display_proxy_subtree(callback=lambda disc: disc.is_configured())

        Arguments:
            callback (method taking ProxyDiscipline as input) : callback function to show for each ProxyDiscipline in []
        """
        proxy_subtree = []
        self.get_proxy_subtree_rec(proxy_subtree, 0, callback)
        return '\n'.join(proxy_subtree)

    def get_proxy_subtree_rec(self, proxy_subtree, indent=0, callback=None):
        callback_string = ' [' + str(callback(self)) + ']' if callback is not None else ''
        proxy_subtree.append(
            '    ' * indent
            + '|_ '
            + self.ee.ns_manager.get_local_namespace_value(self)
            + '  ('
            + self.__class__.__name__
            + ')'
            + callback_string
        )
        for disc in self.proxy_disciplines:
            disc.get_proxy_subtree_rec(proxy_subtree, indent + 1, callback)

    def get_inst_desc_in(self):
        return self.inst_desc_in

    def get_father_executor(self):
        return self.father_executor

    def get_numerical_outputs_for_discipline(self) -> dict:
        """
        Method that returns the numerical outputs in the discipline data out if there is an attribute in the GEMSEO
        discipline object that has the same name, filling with None otherwise.
            Returns: numerical outputs of the discipline
        """
        disc_out = self.get_data_out()
        return {
            self.get_var_full_name(key, disc_out): getattr(self.discipline_wrapp.discipline, key, None)
            for key in disc_out
            if disc_out[key][self.NUMERICAL] is True
        }

    def get_numerical_outputs_subprocess(self) -> dict:
        """
        Method that returns the numerical outputs recursively in the subprocess proxy disciplines.
            Returns: numerical outputs of the discipline and subprocess
        """
        numerical_outputs_subprocess = self.get_numerical_outputs_for_discipline()
        for disc in self.proxy_disciplines:
            numerical_outputs_subprocess.update(disc.get_numerical_outputs_subprocess())
        return numerical_outputs_subprocess
