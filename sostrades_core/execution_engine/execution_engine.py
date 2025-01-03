'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/06-2024/07/30 Copyright 2023 Capgemini

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
# Execution engine SoSTrades code
import logging
from typing import Any, Callable, Optional, Union

from sostrades_core.datasets.dataset_mapping import DatasetsMapping
from sostrades_core.execution_engine.builder_tools.tool_factory import ToolFactory
from sostrades_core.execution_engine.data_manager import DataManager, ParameterChange
from sostrades_core.execution_engine.ns_manager import NamespaceManager
from sostrades_core.execution_engine.post_processing_manager import (
    PostProcessingManager,
)
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling, BaseDiscipline, BaseScenario
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.scattermaps_manager import ScatterMapsManager
from sostrades_core.execution_engine.sos_factory import SosFactory

DEFAULT_FACTORY_NAME = 'default_factory'
DEFAULT_NS_MANAGER_NAME = 'default_ns_namanger'
DEFAULT_scattermap_manager_NAME = 'default_smap_namanger'


class ExecutionEngineException(Exception):
    pass


class ExecutionEngine:
    """SoSTrades execution engine"""
    STUDY_AND_ROOT_PLACEHODER: str = '<study_and_root_ph>'
    STUDY_PLACEHOLDER_WITH_DOT: str = '<study_ph>.'
    STUDY_PLACEHOLDER_WITHOUT_DOT: str = '<study_ph>'

    LOG_LEVEL: int = logging.INFO
    """The default log level (INFO)."""

    def __init__(self, study_name,
                 rw_object=None,
                 root_dir=None,
                 study_filename=None,
                 yield_method=None,
                 logger: Optional[logging.Logger] = None):

        self.study_name = study_name
        self.study_filename = study_filename or study_name
        self.__yield_method = yield_method
        if logger is None:
            # Use rsplit to get sostrades_core.execution_engine instead of sostrades_core.execution_engine.execution_engine
            # as a default logger if not initialized
            self.logger = logging.getLogger(f"{__name__.rsplit('.', 2)[0]}.{self.__class__.__name__}")
        else:
            self.logger = logger
        self.logger.setLevel(self.LOG_LEVEL)

        self.__post_processing_manager = PostProcessingManager(self)

        self.ns_manager = NamespaceManager(name=DEFAULT_NS_MANAGER_NAME, ee=self)
        self.dm = DataManager(name=self.study_name,
                              root_dir=root_dir,
                              rw_object=rw_object,
                              study_filename=self.study_filename,
                              ns_manager=self.ns_manager,
                              logger=self.logger.getChild("DataManager"))
        self.scattermap_manager = ScatterMapsManager(
            name=DEFAULT_scattermap_manager_NAME, ee=self)

        self.tool_factory = ToolFactory(
            self, self.study_name)

        self.__factory = SosFactory(
            self, self.study_name)

        self.root_process: Union[ProxyCoupling, None] = None
        self.root_builder_ist = None
        self.check_data_integrity: bool = True
        self.wrapping_mode = 'SoSTrades'

    @property
    def factory(self) -> SosFactory:
        """ Read-only accessor to the factory object

            :return: current used factory
            :type: SosFactory
        """
        return self.__factory

    @property
    def post_processing_manager(self) -> PostProcessingManager:
        """ Read-only accessor to the post_processing_manager object

            :return: current used post_processing_manager
            :type: PostProcessingManager
        """
        return self.__post_processing_manager

    # -- Public methods
    def select_root_process(self, repo, mod_id):

        # Method usage now ?
        # dead code in comment
        # usage regarding 'select_root_builder_ist'
        # usage only in testing not at runtime
        self.logger.warning(
            'DEPRECATION WARNING (07/2021).\n"select_root_process" methods is flagged to be checked regarding "select_root_builder_ist" method and usage in code (only in testing behaviour)')
        #         builder_list = self.factory.get_builder_from_process(repo=repo,
        #                                                              mod_id=mod_id)
        #         self.factory.set_builders_to_coupling_builder(builder_list)
        #
        #         self.load_study_from_input_dict({})

        # Set main process information to factory
        self.factory.repository = repo
        self.factory.process_identifier = mod_id

        self.select_root_builder_ist(repo, mod_id)
        self.attach_builders_to_root()

    def select_root_builder_ist(self, repo, mod_id):
        self.factory.repository = repo
        self.factory.process_identifier = mod_id
        self.root_builder_ist = self.factory.get_pb_ist_from_process(
            repo, mod_id)

    def attach_builders_to_root(self):
        builder_list_func = getattr(
            self.root_builder_ist, self.factory.BUILDERS_FUNCTION_NAME)
        builder_list = builder_list_func()

        if isinstance(builder_list, ProxyDiscipline.GEMSEO_OBJECTS):
            self.factory.set_gemseo_object_to_coupling_builder(builder_list)
            self.wrapping_mode = 'GEMSEO'
        else:
            self.factory.set_builders_to_coupling_builder(builder_list)
            # -- We are changing what happend in root, need to reset dm
        self.dm.reset()
        self.load_study_from_input_dict({})

    def set_root_process(self, process_instance: ProxyCoupling):
        # self.dm.reset()
        if isinstance(process_instance, ProxyCoupling):
            self.root_process = process_instance
        else:
            raise ExecutionEngineException(
                f'Execution engine root process is intended to be an instance or inherited instance of ProxyDiscipline class and not {type(process_instance)}.')

    def configure(self):
        self.logger.info('configuring ...')
        self.factory.build()
        self.root_process.configure()

        # create DM treenode to be able to populate it from GUI
        self.dm.treeview = None

    def prepare_execution(self):
        '''
        loop on proxy disciplines and execute prepare execution
        '''
        self.logger.info("Preparing execution.")
        # - instantiate models in user wrapps

        # clean unused namespaces
        self.clean_unused_namespaces()

        self.__factory.init_execution()
        # - execution
        self.root_process.prepare_execution()

    def __configure_io(self):
        self.logger.info('Configuring IO...')

        self.factory.build()
        self.root_process.configure_io()

    def update_from_dm(self):
        self.logger.info("Updating from DM.")
        self.root_process.update_from_dm()

    def build_cache_map(self):
        '''
        Build cache map with all gemseo disciplines cache
        '''
        self.dm.reinit_cache_map()

        self.root_process._set_dm_cache_map()

    def anonymize_caches_in_cache_map(self):
        """
        Anonymize each cache in the cache map by converting each variable key inside the cache
        to its anonymized form. The returned cache map is a dictionary with the same structure,
        but with anonymized keys.

        Returns:
        - dict: A dictionary with the same structure as cache_map, but with anonymized keys.
        """

        def anonymize_dict(dict_to_anonymize):
            """
            Convert each key in the dictionary to its anonymized form.

            Parameters:
            - dict_to_anonymize (dict): A dictionary with original keys.

            Returns:
            - dict: A dictionary with anonymized keys.
            """
            return {self.anonymize_key(key): value for key, value in dict_to_anonymize.items()}

        def anonymize_dict_of_dict(dict_to_anonymize):
            """
            Convert each key in a dictionary of dictionary to its anonymized form.

            Parameters:
            - dict_to_anonymize (dict): A dictionary with original keys.

            Returns:
            - dict: A dictionary with anonymized keys.
            """
            return {self.anonymize_key(key): anonymize_dict(value) for key, value in dict_to_anonymize.items()}

        # Initialize an empty dictionary to store the anonymized cache map
        anonymized_cache_map = {}

        # If the cache map is not empty, process each entry
        if self.dm.cache_map:
            for key, serialized_new_cache in self.dm.cache_map.items():
                anonymized_cache = {}
                for index, cache in enumerate(serialized_new_cache, start=1):
                    # Anonymize the keys for inputs, outputs, and jacobian of each cache
                    anonymized_cache[index] = {
                        'inputs': anonymize_dict(cache.inputs),
                        'outputs': anonymize_dict(cache.outputs),
                        'jacobian': anonymize_dict_of_dict(cache.jacobian)
                    }
                # Add the anonymized cache to the result dictionary
                anonymized_cache_map[key] = anonymized_cache

        return anonymized_cache_map

    def unanonymize_caches_in_cache_map(self, cache_map):
        """
        Unanonymize each cache in the cache map by converting each key inside the cache
        to its original form. The returned cache map is a dictionary with the same structure,
        but with unanonymized keys.

        Parameters:
        - cache_map (dict): A dictionary where each value is a list of cache objects.

        Returns:
        - dict: A dictionary with the same structure as cache_map, but with unanonymized keys.
        """

        def un_anonymize_dict(dict_to_anonymize):
            """
            Convert each key in the dictionary to its original form.

            Parameters:
            - dict_to_anonymize (dict): A dictionary with anonymized keys.

            Returns:
            - dict: A dictionary with the original keys.
            """
            return {self.__unanonimize_key(key): value for key, value in dict_to_anonymize.items()}

        def un_anonymize_dict_of_dict(dict_to_anonymize):
            """
            Convert each key in the dictionary to its original form.

            Parameters:
            - dict_to_anonymize (dict): A dictionary with anonymized keys.

            Returns:
            - dict: A dictionary with the original keys.
            """
            return {self.__unanonimize_key(key): un_anonymize_dict(value) for key, value in dict_to_anonymize.items()}

        # Initialize an empty dictionary to store the unanonymized cache map

        unanonymized_cache_map = {}
        cache_unanonymize_func = {'inputs': un_anonymize_dict, 'outputs': un_anonymize_dict,
                                  'jacobian': un_anonymize_dict_of_dict}
        # If the cache map is not empty, process each entry
        if cache_map:
            # Unanonymize the keys for inputs, outputs, and jacobian of each cache
            unanonymized_cache_map = {
                key: {index: {cache_type: cache_func(cache[cache_type]) for cache_type, cache_func in
                              cache_unanonymize_func.items()} for index, cache in serialized_cache.items()} for
                key, serialized_cache in cache_map.items()}

        return unanonymized_cache_map

    def get_cache_map_to_dump(self):
        '''
        Build if necessary and return data manager cache map
        '''
        if self.dm.cache_map is None:
            self.build_cache_map()

        anonymized_cache_map = self.anonymize_caches_in_cache_map()

        return anonymized_cache_map

    def load_cache_from_map(self, cache_map):
        '''
        Load disciplines cache from cache_map
        '''

        if len(cache_map) > 0:
            # build cache map and gemseo disciplines id map in data manager
            self.build_cache_map()

            unanonymized_cache_map = self.unanonymize_caches_in_cache_map(
                cache_map)
            # store cache of all gemseo disciplines

            self.dm.fill_gemseo_caches_with_unanonymized_cache_map(
                unanonymized_cache_map)

    def update_status_configure(self):
        '''
        Update status configure of all disciplines in factory
        '''
        for disc in self.factory.proxy_disciplines:
            disc._update_status_dm(ProxyDiscipline.STATUS_CONFIGURE)

    def get_treeview(self, no_data=False, read_only=False, exec_display=False):
        ''' returns the treenode build based on datamanager '''
        if self.dm.treeview is None or self.dm.treeview.exec_display != exec_display:
            self.dm.create_treeview(
                self.root_process, self.__factory.process_module, no_data, read_only, exec_display=exec_display)

        return self.dm.treeview

    def display_treeview_nodes(self, display_variables=None, exec_display=False):
        '''
        Display the treeview and create it if not
        '''
        self.get_treeview(exec_display=exec_display)
        tv_to_display = self.dm.treeview.display_nodes(
            display_variables=display_variables)
        self.logger.info(tv_to_display)
        return tv_to_display

    def anonymize_key(self, key_to_anonymize):
        base_namespace = f'{self.study_name}.{self.root_process.sos_name}'
        converted_key = key_to_anonymize

        if key_to_anonymize == base_namespace:
            converted_key = key_to_anonymize.replace(
                base_namespace, ExecutionEngine.STUDY_AND_ROOT_PLACEHODER, 1)

        elif key_to_anonymize.startswith(f'{base_namespace}.'):
            converted_key = key_to_anonymize.replace(
                f'{base_namespace}.', f'{ExecutionEngine.STUDY_AND_ROOT_PLACEHODER}.', 1)

        elif key_to_anonymize.startswith(f'{self.study_name}.'):
            converted_key = key_to_anonymize.replace(
                f'{self.study_name}.', ExecutionEngine.STUDY_PLACEHOLDER_WITH_DOT, 1)
        elif key_to_anonymize.startswith(f'{self.study_name}'):
            converted_key = key_to_anonymize.replace(
                f'{self.study_name}', ExecutionEngine.STUDY_PLACEHOLDER_WITHOUT_DOT, 1)

        return converted_key

    def __unanonimize_key(self, key_to_unanonimize):
        base_namespace = f'{self.study_name}.{self.root_process.sos_name}'
        converted_key = key_to_unanonimize

        if key_to_unanonimize == ExecutionEngine.STUDY_AND_ROOT_PLACEHODER:
            converted_key = key_to_unanonimize.replace(
                ExecutionEngine.STUDY_AND_ROOT_PLACEHODER, base_namespace)

        elif key_to_unanonimize.startswith(f'{ExecutionEngine.STUDY_AND_ROOT_PLACEHODER}.'):
            converted_key = key_to_unanonimize.replace(
                f'{ExecutionEngine.STUDY_AND_ROOT_PLACEHODER}.', f'{base_namespace}.')

        elif key_to_unanonimize.startswith(ExecutionEngine.STUDY_PLACEHOLDER_WITH_DOT):
            converted_key = key_to_unanonimize.replace(
                ExecutionEngine.STUDY_PLACEHOLDER_WITH_DOT, f'{self.study_name}.')
        elif key_to_unanonimize.startswith(ExecutionEngine.STUDY_PLACEHOLDER_WITHOUT_DOT):
            converted_key = key_to_unanonimize.replace(
                ExecutionEngine.STUDY_PLACEHOLDER_WITHOUT_DOT, f'{self.study_name}')

        return converted_key

    def export_data_dict_and_zip(self, export_dir):
        '''
        serialise data dict of the study keeping namespaced variables
        and generate csv files of whole data
        '''
        self.logger.info('exporting data from study to externalised files...')
        self.logger.debug('dumping study before...')
        return self.dm.export_data_dict_and_zip(export_dir)

    def load_disciplines_status_dict(self, disciplines_status_dict):
        '''
        Read disciplines status dict given as argument and then update the execution engine
        accordingly

        :params: disciplines_status_dict, dictionary {disciplines_key: status}
        '''
        for discipline_key in self.dm.disciplines_dict:

            dm_discipline = self.dm.disciplines_dict[discipline_key][DataManager.DISC_REF]

            # Get basic discipline key
            target_key = dm_discipline.get_disc_full_name()

            # Check if basic key is available in the discipline status
            # dictionary to load
            if target_key not in disciplines_status_dict:
                # If not convert basic key to an anonimized key
                target_key = self.anonymize_key(target_key)

            if target_key in disciplines_status_dict:
                status_to_load = disciplines_status_dict[target_key]

                if isinstance(status_to_load, dict):
                    if self.dm.disciplines_dict[discipline_key]['classname'] in status_to_load:
                        status = status_to_load[self.dm.disciplines_dict[discipline_key]['classname']]
                        self.dm.disciplines_dict[discipline_key]['status'] = status
                        dm_discipline.status = status
                else:
                    self.dm.disciplines_dict[discipline_key]['status'] = status_to_load
                    dm_discipline.status = status_to_load

    def get_anonimated_disciplines_status_dict(self):
        '''
        Return the execution engine discipline status dictionary but with anonimize key

        :returns: dictionary {anonimize_disciplines_key: status}
        '''
        converted_dict = {}
        dict_to_convert = self.dm.build_disc_status_dict()

        for discipline_key in dict_to_convert.keys():
            converted_key = self.anonymize_key(discipline_key)
            converted_dict[converted_key] = dict_to_convert[discipline_key]

        return converted_dict

    def load_study_from_dataset(self, datasets_mapping: DatasetsMapping, update_status_configure: bool = True):
        '''
        Load a study from a datasets mapping dictionary : retreive dataset value and load study

        :param datasets_mapping: Dataset mapping to use
        :type datasets_mapping: DatasetsMapping
        :param update_status_configure: whether to update the status for configure
        :type update_status_configure: bool
        '''
        # call the configure function with the set dm data from datasets
        return self.configure_study_with_data(datasets_mapping, self.dm.fill_data_dict_from_datasets,
                                              update_status_configure)

    def load_study_from_input_dict(self, input_dict_to_load, update_status_configure=True):
        '''
        Load a study from an input dictionary : Convert the input_dictionary into a dm-like dictionary
        and compute the function load_study_from_dict
        '''
        dict_to_load = self.convert_input_dict_into_dict(input_dict_to_load)
        return self.load_study_from_dict(dict_to_load, self.__unanonimize_key,
                                         update_status_configure=update_status_configure)

    def get_anonimated_data_dict(self):
        '''
        return execution engine data dict using anonimizin key for serialisation purpose
        '''

        converted_dict = {}
        dict_to_convert = self.dm.convert_data_dict_with_full_name()

        for key in dict_to_convert.keys():
            new_key = self.anonymize_key(key)
            converted_dict[new_key] = dict_to_convert[key]

        return converted_dict

    def convert_input_dict_into_dict(self, input_dict):

        dm_dict = {key: {ProxyDiscipline.VALUE: value} for key, value in input_dict.items()}
        return dm_dict

    def load_study_from_dict(
            self, dict_to_load: dict[str:Any], anonymize_function=None, update_status_configure: bool = True
    ):
        '''
        method that imports data from dictionary to discipline tree

        :params: anonymize_function, a function that map a given key of the data
        dictionary using rule given by the execution engine for the saving process.
        If provided this function is use to load an anonymized reference data and
        compare the key with those of the current process
        :type: function

        :params: target_disc, SoSEval discipline to be configured during run method
        Optional parameter used only for evaluator process to avoid the configuration of all disciplines
        :type: SoSEval object
        '''
        self.logger.debug("Loading study from dictionary")

        if anonymize_function is None:
            data_cache = dict_to_load
        else:
            data_cache = {}
            for key, value in dict_to_load.items():
                converted_key = anonymize_function(key)
                data_cache.update({converted_key: value})

        # call the configure function with the set dm data from dict
        return self.configure_study_with_data(data_cache, self.dm.fill_data_dict_from_dict, update_status_configure)

    def configure_study_with_data(
            self,
            dict_or_datasets_to_load: Union[dict, DatasetsMapping],
            set_data_in_dm_function: Callable[
                [Union[dict, DatasetsMapping], set[str], list[ParameterChange], bool, bool, bool], None],
            update_status_configure: bool,

    ):
        '''
        method that insert data into dm and configure the process

        :param set_data_in_dm_function: a function sets data in datamanager data_dict using dict_or_datasets_to_load, with signature:
        set_data_in_dm_function(dict_or_datasets_to_load:Union[dict, DatasetsMapping], already_inserted_keys: set of data name, in_vars:bool, init_coupling_vars:bool, out_vars:bool) -> None
        :type set_data_in_dm_function: Callable

        '''
        iteration = 0

        loop_stop = False
        # convergence loop: run discipline configuration until the number of sub disciplines is stable
        # that should mean all disciplines under discipline to load are deeply
        # configured

        checked_keys = set()
        parameter_changes = []

        while not loop_stop:
            self.logger.info("Configuring loop iteration %i.", iteration)
            if self.__yield_method is not None:
                self.__yield_method()

            self.dm.no_change = True
            # call the function that will set data in dm
            set_data_in_dm_function(dict_or_datasets_to_load, checked_keys, parameter_changes, in_vars=True,
                                    init_coupling_vars=False, out_vars=False)

            self.__configure_io()

            if self.__yield_method is not None:
                self.__yield_method()

            iteration = iteration + 1

            if self.root_process.is_configured():
                loop_stop = True
            elif iteration >= 100:
                self.logger.warning('CONFIGURE WARNING: root process is not configured after 100 iterations')
                raise ExecutionEngineException('Too many iterations')

        # Convergence is ended
        # Set all output variables and strong couplings

        set_data_in_dm_function(dict_or_datasets_to_load, checked_keys, parameter_changes, in_vars=False,
                                init_coupling_vars=True, out_vars=True)

        if self.__yield_method is not None:
            self.__yield_method()

        # -- Init execute, to fully initialize models in discipline
        if len(parameter_changes) > 0:
            self.update_from_dm()
            if update_status_configure:
                self.update_status_configure()
        else:
            if self.dm.treeview is not None:
                self.root_process.status = self.dm.treeview.root.status

        # Reduced dm recreation might be necessary without value change (i.e. a type changed during config.)
        if checked_keys or self.dm.reduced_dm is None:
            self.dm.create_reduced_dm()

        self.dm.treeview = None
        return parameter_changes

    def clean_unused_namespaces(self):
        '''

        Returns:

        '''
        post_processing_ns_list = list(self.post_processing_manager.namespace_post_processing.keys())
        self.ns_manager.clean_unused_namespaces(post_processing_ns_list)

    def get_data_integrity_msg(self) -> str:
        """gathers the messages concerning data integrity"""
        integrity_msg_list = [
            f'Variable {self.dm.get_var_full_name(var_id)} : {var_data_dict[ProxyDiscipline.CHECK_INTEGRITY_MSG]}'
            for var_id, var_data_dict in self.dm.data_dict.items() if
            var_data_dict[ProxyDiscipline.CHECK_INTEGRITY_MSG]]

        full_integrity_msg = '\n'.join(integrity_msg_list)
        return full_integrity_msg

    def __check_data_integrity_msg(self, raise_exceptions: bool = True):
        '''
        Check if one data integrity msg is not empty string to crash a value error
        as the old check_inputs in the dm juste before the execution
        Add the name of the variable in the message
        '''
        data_integrity_msg = self.get_data_integrity_msg()
        if data_integrity_msg:
            raise ValueError(data_integrity_msg)

    def set_debug_mode(self, mode=None, disc=None):
        ''' set recursively <disc> debug options of in ProxyDiscipline
        '''
        # TODO : update with new debug mode logic
        if disc is None:
            disc = self.root_process
        mode_str = mode
        if mode_str is None:
            mode_str = "all"
        msg = "Debug mode activated for discipline %s with mode <%s>" % (disc.get_disc_full_name(), mode_str)
        self.logger.info(msg)
        # set check options
        if mode is None:
            disc.nan_check = True
            disc.check_if_input_change_after_run = True
            disc.check_min_max_couplings = True
        elif mode == "nan":
            disc.nan_check = True
        elif mode == "input_change":
            disc.check_if_input_change_after_run = True
        # we deactivate these debug mode for gemseo convergence to start, need to overload some methods if needed
        # elif mode == "linearize_data_change":
        #     disc.check_linearize_data_changes = True
        # elif mode == "min_max_grad":
        #     disc.check_min_max_gradients = True
        elif mode == "min_max_couplings":
            if isinstance(disc, ProxyCoupling):
                for sub_mda in disc.inner_mdas:
                    sub_mda.debug_mode_couplings = True
        elif mode == 'data_check_integrity':
            self.check_data_integrity = True

        else:
            avail_debug = ["nan", "input_change", "min_max_couplings", 'data_check_integrity']
            raise ValueError("Debug mode %s is not among %s" % (mode, str(avail_debug)))
        # set debug modes of subdisciplines
        for disc in disc.proxy_disciplines:
            self.set_debug_mode(mode, disc)

    def get_input_data_for_gemseo(self, proxy_coupling):
        '''
        Get values of discipline input_grammar from data manager
        '''
        input_data = {}
        input_data_names = proxy_coupling.discipline_wrapp.discipline.input_grammar.names
        if len(input_data_names) > 0:
            for data_name in input_data_names:
                input_data[data_name] = self.dm.get_value(data_name)

        return input_data

    def update_dm_with_local_data(self, local_data):
        '''
        Update the DM with local data from GEMSEO

        Arguments:
            local_data (dict): to update datamanager with
        '''
        self.dm.set_values_from_dict(local_data)

    def execute(self, loaded_cache=None):
        ''' execution of the execution engine
        '''
        self.logger.info('PROCESS EXECUTION %s STARTS...', self.root_process.get_disc_full_name())
        #         self.root_process.clear_cache()
        self.update_from_dm()

        self.__check_data_integrity_msg()

        # -- prepare execution
        self.prepare_execution()

        if loaded_cache is not None:
            self.load_cache_from_map(loaded_cache)

        # -- execution with input data from DM
        ex_proc = self.root_process
        input_data = self.dm.get_data_dict_values(excepted=['numerical'])
        self.logger.info("Executing.")
        input_data_wo_none = {key: value for key, value in input_data.items() if value is not None}
        try:
            if self.wrapping_mode == 'SoSTrades':
                ex_proc.discipline_wrapp.discipline.execute(
                input_data=input_data_wo_none)
                io_data = ex_proc.discipline_wrapp.discipline.io.data
            elif self.wrapping_mode == 'GEMSEO':
                ex_proc.discipline_wrapp.discipline.execute()
                if isinstance(ex_proc.discipline_wrapp.discipline, BaseDiscipline):
                    gemseo_disc = ex_proc.discipline_wrapp.discipline
                    io_data = {f'{self.study_name}.{key}': value for key, value in gemseo_disc.io.data.items()}
                elif isinstance(ex_proc.discipline_wrapp.discipline, BaseScenario):
                    gemseo_discs = ex_proc.discipline_wrapp.discipline.disciplines
                    io_data = {}
                    for gemseo_disc in gemseo_discs:
                        io_data.update(
                            {f'{self.study_name}.{key}': value for key, value in gemseo_disc.io.data.items()})

        except:
            ex_proc.set_status_from_discipline()
            raise

        self.status = self.root_process.status
        self.logger.info('PROCESS EXECUTION %s ENDS.', self.root_process.get_disc_full_name())

        if self.wrapping_mode == 'SoSTrades':
            self.logger.info("Storing local data in datamanager.")
            # -- store local data in datamanager
            io_data.pop("MDA residuals norm", None)
        self.update_dm_with_local_data(io_data)
        # Add residuals_history and other numerical outputs that are not in GEMSEO grammar to the data manager
        self.update_dm_with_local_data(ex_proc.get_numerical_outputs_subprocess())
        # -- update all proxy statuses
        ex_proc.set_status_from_discipline()

        return ex_proc
