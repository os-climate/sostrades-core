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
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Class that manage a whole study process (load, execute, save, dump..)
"""

from time import time
from os.path import join, isdir
from logging import INFO, DEBUG

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from sos_trades_core.tools.tree.serializer import DataSerializer
from sos_trades_core.tools.rw.load_dump_dm_data import DirectLoadDump, AbstractLoadDump
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from copy import deepcopy
from gemseo.utils.compare_data_manager_tooling import compare_dict

# CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG
LOG_LEVEL = INFO  # = 20

# Pylint code disable section
# pylint: disable=line-too-long, logging-format-interpolation


class BaseStudyManager():
    """ Class defninition

        Base class use to manage making, loading and saving data for a process into an execution engine instance

        redefining the method 'setup_use_case' allow to change the way to load data into the execution engine
    """

    def __init__(self, repository_name, process_name, study_name, dump_directory=None, run_usecase=True, yield_method=None, logger=None, execution_engine=None):
        """ Constructor

        :params: repository_name, package name that contain the target process to load
        :type: str

        :params: process_name, process name of the target process to load
        :type: str

        :params: study_name, study name
        :type: str
        """
        self.__run_usecase = run_usecase
        self.study_name = study_name
        self.repository_name = repository_name
        self.process_name = process_name
        self.dump_directory = dump_directory
        self.__logger = logger
        self.__execution_engine = None
        self.__rw_strategy = DirectLoadDump()
        self.__yield_method = yield_method
        self.__execution_engine = execution_engine

    def run_usecase(self):
        return self.__run_usecase

    def set_run_usecase(self, run_usecase):
        self.__run_usecase = run_usecase

    @property
    def ee(self):
        """ Return the current execution engine instance

        :return: sos_trades_core.execution_engine.execution_engine.ExecutionEngine
        """
        return self.execution_engine

    @property
    def execution_engine(self):
        """ Return the current execution engine instance

        :return: sos_trades_core.execution_engine.execution_engine.ExecutionEngine
        """
        if self.__execution_engine is None:
            self._init_exec_engine()

        return self.__execution_engine

    @property
    def rw_strategy(self):
        """ Get the read/write strategy used for loading and dumping execution engine data
        """
        return self.__rw_strategy

    @rw_strategy.setter
    def rw_strategy(self, rw_strategy):
        """ Set a new read/write strategy for loading and dumping execution engine data
        """
        if rw_strategy is not None:
            self.__rw_strategy = rw_strategy

    @property
    def logger(self):
        """ return the current Study logger object

        @return logging.logger
        """
        return self.__logger

    @property
    def study_data_file_path(self):
        serializer = DataSerializer()
        serializer.set_dm_pkl_files(study_to_load=self.dump_directory)

        return serializer.dm_pkl_file

    def _init_exec_engine(self):
        """ Create an isntance of the execution engine
        """
        self.__execution_engine = ExecutionEngine(
            self.study_name, root_dir=self.dump_directory, yield_method=self.__yield_method, logger=self.__logger)

        # set the level of ExecutioEngine logger and all others its children
        self.__execution_engine.logger.setLevel(LOG_LEVEL)

        self.ee.select_root_builder_ist(
            self.repository_name, self.process_name)
        self.setup_process()
        self.ee.attach_builders_to_root()

    def setup_process(self):
        pass

    def load_data(self, from_path=None, from_input_dict=None, display_treeview=True, from_connectors_dict=None):
        """ Method that load data into the execution engine

        :params: from_path, location of pickle file to load (optional parameter)
        :type: str

        :params: from_input_dict, input dict with (optional parameter), be carreful,
                it break the call to 'setup_usecase' method
        :type: str

        :params: display_treeview, display or not treeview state (optional parameter)
        :type: boolean
        """
        start_time = time()

        logger = self.execution_engine.logger

        if display_treeview:
            logger.info('TreeView display BEFORE data setup & configure')
            self.execution_engine.display_treeview_nodes()

        # Retrieve data to load and make sure they have the correct type
        usecase_data = []

        if from_input_dict is not None:
            usecase_data = from_input_dict
        elif from_path is None:
            usecase_data = self.setup_usecase()
        else:
            usecase_data = self.setup_usecase(
                study_folder_path=from_path)

        if not isinstance(usecase_data, list):
            usecase_data = [usecase_data]
        input_dict_to_load = {}

        for uc_d in usecase_data:
            input_dict_to_load.update(uc_d)

        # Initialize execution engine with data
        # import ipdb
        # ipdb.set_trace()
        self.execution_engine.load_study_from_input_dict(input_dict_to_load)
        self.load_connectors(
            from_dict=from_connectors_dict, from_path=from_path)
        self.specific_check()
        if display_treeview:
            logger.info('TreeView display AFTER  data setup & configure')
            self.execution_engine.display_treeview_nodes()

        study_display_name = f'{self.repository_name}.{self.process_name}.{self.study_name}'
        message = f'Study {study_display_name} loading time : {time() - start_time} seconds'
        logger.info(message)

    def specific_check(self):
        """ Method to overload to have a specific check on input datas

        """
        pass

    def load_connectors(self, from_dict=None, from_path=None):
        """Method that load connectors into the execution engine

        :params: from_dict, connectors to save in dm
        :type: dict

        :params: from_path, study folder path to get connectors into file and to set them in dm
        :type: str
        """
        connectors_dict = {}
        if from_dict is not None:
            # connector list is given in input
            connectors_dict = from_dict
        else:
            # if there is no from path, we get the data from dump directory
            if from_path is None and self.dump_directory is not None and isdir(self.dump_directory):
                from_path = self.dump_directory
            # else we get it from the path in input

            if from_path is not None:
                # get connectors from file data
                serializer = DataSerializer(root_dir=from_path)
                loaded_dict = serializer.get_dict_from_study(
                    from_path, self.__rw_strategy)
                for key, param_data in loaded_dict.items():
                    if SoSDiscipline.CONNECTOR_DATA in param_data.keys():
                        connectors_dict[key] = param_data[SoSDiscipline.CONNECTOR_DATA]

        if len(connectors_dict) > 0:
            self.execution_engine.load_connectors_from_dict(connectors_dict)

    def dump_data(self, study_folder_path):
        """ Method that dump data from the execution engine to a file

        :params: study_folder_path, location of pickle file to load
        :type: str
        """

        # Retrieve data to dump
        data = self.execution_engine.get_anonimated_data_dict()

        self._put_data_into_file(study_folder_path, data)

    def load_disciplines_data(self, study_folder_path=None):
        """ Method that load data into the execution engine

        :params: study_folder_path, location of pickle file to load (optional parameter)
        :type: str
        """

        # Retrieve data to load and make sure they have the correct type
        loaded_dict = self.setup_disciplines_data(study_folder_path)

        # Initialize execution engine with data
        self.execution_engine.load_disciplines_status_dict(
            loaded_dict)

    def dump_disciplines_data(self, study_folder_path):
        """ Method that load data into the execution engine

        :params: study_folder_path, location of pickle file to load
        :type: str
        """

        # Retrieve data to dump
        data = self.execution_engine.get_anonimated_disciplines_status_dict()

        self._put_disciplines_data_into_file(study_folder_path, data)

    def run(self, logger_level=None,
            dump_study=False,
            for_test=False):
        """ Method that run execution engine study with some additionals options

        :params: logger_level, target logging level request for the run (None by default,
                    so use the current class variable value => INFO)
        :type: str/int (logging level constant)
        """
        # import ipdb
        # ipdb.set_trace()
        logger = self.execution_engine.logger

        # Manage logging level request in arguments
        # Make some check in order to make sure we can set it into the logging
        # API
        if logger_level is not None:
            log_lvl = DEBUG
            try:
                if isinstance(logger_level, str):
                    # try to map with logging levels
                    log_lvl = eval(logger_level.upper())
                else:
                    # try to convert in integer to match with logging level
                    # types
                    log_lvl = int(logger_level)
            except:
                pass
            logger.setLevel(log_lvl)
            logger.info(
                f'set logger level to {log_lvl}')

        # Do not display information on process location on standard run (not
        # DEBUG)
        study_display_name = ''
        if logger.level == DEBUG:
            study_display_name = f'{self.repository_name}.{self.process_name}.{self.study_name}'
        else:
            study_display_name = self.study_name

        logger.info(f'Study {study_display_name} starts...')

        # Execute study
        start_time = time()
        if self.__run_usecase:
            try:
                self.execution_engine.execute()
                message = f'Study {study_display_name} execution time : {time() - start_time} seconds'
                logger.info(message)
                print(message)
                if for_test:
                    self.__launch_additional_test()
            except Exception as ex:
                message = f'Study {study_display_name} execution time on error : {time() - start_time} seconds'
                logger.info(message)
                print(message)
                raise ex

        else:
            print(f'Study {study_display_name} is configured not to run.')
            print(f'Skipping execute.')
            logger.info(f'Study {study_display_name} is configured not to run.')
            logger.info(f'Skipping execute.')

        # Method after execute and before dump
        try:
            self.after_execute_before_dump()
        except Exception:
            logger.exception(
                'The following error occurs in "after_execute_before_dump" methods')

        if dump_study and self.dump_directory is not None:
            self.dump_data(self.dump_directory)
            self.dump_disciplines_data(self.dump_directory)
            logger.debug(
                f'Reference dump to {self.dump_directory}')

        logger.info(f'Study {study_display_name} done.')

    def setup_usecase(self, study_folder_path=None):
        """ Method to overload in order to provide data to the loaded study process
        from a specific way

        :params: study_folder_path, location of pickle file to load (optional parameter)
        :type: str

        :return: list od dictionary, [{str: *}]
        """

        if study_folder_path is not None and isdir(study_folder_path):
            return self._get_data_from_file(study_folder_path)

        return []

    def setup_disciplines_data(self, study_folder_path=None):
        """ Method to overload in order to provide data to the loaded study process
        from a specific way

        :params: study_folder_path, location of pickle file to load (optional parameter)
        :type: str

        :return: dictionary, {str: *}
        """

        if study_folder_path is not None and isdir(study_folder_path):
            return self._get_disciplines_data_from_file(study_folder_path)

        return {}

    def after_execute_before_dump(self):
        ''' treatment before dumping in order to remove/change data before dumping '''

    def __launch_additional_test(self):
        """ Launch additional test based on execution engine

        """
        logger = self.execution_engine.logger

        # Save data manager before post-processing
        dm_dict_before = deepcopy(
            self.execution_engine.get_anonimated_data_dict())

        logger.info(
            f'------ Check post-processing integrity ------')
        ppf = PostProcessingFactory()
        ppf.get_all_post_processings(
            execution_engine=self.execution_engine, filters_only=False, for_test=True)

        logger.info(
            f'------ Check data manager integrity after post-processing calls ------')

        # Save data manager after post-processing
        dm_dict_after = deepcopy(
            self.execution_engine.get_anonimated_data_dict())

        test_passed = True
        output_error = ''
        try:
            dict_error = {}
            compare_dict(dm_dict_before, dm_dict_after, '', dict_error)
            if dict_error != {}:
                test_passed = False
                output_error += f'Error while checking data manager after post-processing for usecase {self.repository_name}.{self.process_name}\n'
                for error in dict_error:
                    output_error += f'Mismatch in {error}: {dict_error.get(error)}\n'
                output_error += '---------------------------------------------------------\n'
        except Exception as e:
            test_passed = False
            output_error += f'Error while checking data manager after post-processing for usecase {self.repository_name}.{self.process_name}\n'
            output_error += f'{e}'
            output_error += '\n---------------------------------------------------------\n'

        if not test_passed:
            raise Exception(output_error)

    def set_dump_directory(self, dump_dir):
        """ Method to set the dump directory of the StudyManager

        :params: dump_dir, dump directory
        :type: str
        """
        logger = self.execution_engine.logger

        built_directory = join(
            dump_dir, self.repository_name, self.process_name, self.study_name)

        logger.debug(
            f'Dump directory set to {built_directory}')

        self.dump_directory = built_directory

    def _get_data_from_file(self, study_folder_path):
        """ Method that load data from a file using an serializer object strategy (set with the according setter)

        :params: study_folder_path, location of pickle file to load
        :type: str

        :return: list of dictionary, [{str: *}]
        """
        result = []

        if study_folder_path is not None:
            serializer = DataSerializer()

            loaded_dict = serializer.get_dict_from_study(
                study_folder_path, self.__rw_strategy)

            input_dict = {key: value[SoSDiscipline.VALUE]
                          for key, value in loaded_dict.items()}

        result.append(input_dict)

        return result

    def _put_data_into_file(self, study_folder_path, data):
        """ Method that load save from a file using an serializer object strategy (set with the according setter)
        File will be entirely overwrittent

        :params: study_folder_path, location of pickle file to save
        :type: str

        :params: data, data to save
        :type: dict

        """

        if study_folder_path is not None:
            serializer = DataSerializer()

            serializer.put_dict_from_study(
                study_folder_path, self.__rw_strategy, data)

    def _get_disciplines_data_from_file(self, study_folder_path):
        """ Method that load discipline data into the execution engine

        :params: study_folder_path, location of pickle file to load
        :type: str
        """
        result = {}

        serializer = DataSerializer()

        if study_folder_path is not None:

            result = serializer.load_disc_status_dict(
                study_folder_path, self.__rw_strategy)

        return result

    def _put_disciplines_data_into_file(self, study_folder_path, disciplines_data):
        """ Method that load discipline data into the execution engine

        :params: study_folder_path, location of pickle file to load
        :type: str

        :params: disciplines_data, disciplines data to save
        :type: dict
        """

        if study_folder_path is not None:
            serializer = DataSerializer()

            serializer.dump_disc_status_dict(
                study_folder_path, self.__rw_strategy, disciplines_data)

    @staticmethod
    def static_dump_data(study_folder_path, execution_engine, rw_strategy):
        """ Method that dump the entire execution engine information

        :params: study_folder_path, location of pickle file to load
        :type: str

        :params: execution_engine, execution engine to dump
        :type: sos_trades_core.execution_engine.execution_engine.ExecutionEngine

        :params: rw_strategy, raed/write execution strategy instance
        :type: sos_trades_core.tools.rw.load_dump_dm_data.AbstractLoadDump base type
        """

        if not isinstance(rw_strategy, AbstractLoadDump):
            raise TypeError(
                f'rw_strategy arguments is not an inherited type of AbstractLoadDump')

        if study_folder_path is not None:
            serializer = DataSerializer()

            serializer.put_dict_from_study(
                study_folder_path, rw_strategy, execution_engine.get_anonimated_data_dict())

    @staticmethod
    def static_load_data(study_folder_path, execution_engine, rw_strategy):
        """ Method that load the entire execution engine information

        :params: study_folder_path, location of pickle file to load
        :type: str

        :params: execution_engine, execution engine to dump
        :type: sos_trades_core.execution_engine.execution_engine.ExecutionEngine

        :params: rw_strategy, raed/write execution strategy instance
        :type: sos_trades_core.tools.rw.load_dump_dm_data.AbstractLoadDump base type
        """

        if not isinstance(rw_strategy, AbstractLoadDump):
            raise TypeError(
                f'rw_strategy arguments is not an inherited type of AbstractLoadDump')

        if isdir(study_folder_path):
            serializer = DataSerializer()

            loaded_dict = serializer.get_dict_from_study(
                study_folder_path, DirectLoadDump())

            input_dict = {key: value[SoSDiscipline.VALUE]
                          for key, value in loaded_dict.items()}

            execution_engine.load_study_from_input_dict(input_dict)
