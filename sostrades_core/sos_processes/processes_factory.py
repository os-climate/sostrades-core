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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
#-- process configuration class
from importlib import import_module
from pathlib import Path
from os.path import dirname, relpath, join
from os import environ, sep, pathsep
import yaml

from sostrades_core.api import get_sos_logger

BUILDERS_MODULE_NAME = 'process'
PROCESSES_MODULE_NAME = 'sos_processes'
DEFAULT_RIGHTS_FILE_NAME = 'default_process_rights.yaml'
USER_MAIL = 'user-mail'
GROUP_NAME = 'group-name'

class SoSProcessFactory:
    '''Class to manager processes
    '''

    def __init__(self, additional_repository_list=None, search_python_path=True, logger=None):
        """ SoSProcessFactory constructor

        :params: additional_repository_list, list with additonal repository to load
        :type: list of string, default None

        :params: search_python_path, look for process into python path library or not
        :type: boolean, default True
        """

        self.__processes_dict = None

        # Setup the logging object
        if logger is None:
            self.logger = get_sos_logger('SoS.EE.ProcessFactory')
        else:
            self.logger = logger

        # raw repository list is the one that contain module path to
        # 'PROCESSES_MODULE_NAME'
        self.__raw_repository_list = []

        # repository list is the one that contain module path that contain the
        # 'BUILDERS_MODULE_NAME'
        self.__repository_list = []
        
        # repository file for default process rights location by repository
        self.__process_default_right_files = {}
        self.__user_default_rights_dict = {}
        self.__group_default_rights_dict = {}

        if additional_repository_list is not None and isinstance(additional_repository_list, list):
            self.__raw_repository_list.extend(additional_repository_list)

        if search_python_path:
            self.__add_python_path_processes()

        self._set_processes_dict()
        
        # Set all the default rights in the dicts for each process
        self._set_processes_rights_from_file_dict()

    @property
    def processes_dict(self):
        return self.__processes_dict

    def get_repo_list(self):
        ''' return list of dict {repo name: repo path} '''

        # get all processes from appli root directory
        return self.__repository_list

    def get_processes_id_list(self, repo):
        return self.__processes_dict[repo]

    def get_processes_dict(self):
        """
        Return the buit dictionary processes base on repository list
        """

        return self.__processes_dict
    
    def get_user_default_rights_dict(self):
        """
        Return the buit dictionary processes user default rights base on repository list
        """
        return self.__user_default_rights_dict
    
    def get_group_default_rights_dict(self):
        """
        Return the buit dictionary processes group default rights base on repository list
        """
        return self.__group_default_rights_dict
    
    #-- Protected methods
    def _set_processes_dict(self):
        ''' load processes list
        '''
        #-- re-initialize processes_list
        self.__processes_dict = {}
        self.__repository_list = []

        #-- Set one dict per repo
        for repo_path in self.__raw_repository_list:

            resolve_raw_repository_processes = self.__get_repositories_by_process(
                repo_path)

            self.__repository_list.extend(
                resolve_raw_repository_processes.keys())
            self.__processes_dict.update(resolve_raw_repository_processes)
    
    
    def _set_processes_rights_from_file_dict(self):
        '''
        Retreive the list of process modules
        store them in 2 dictionaries one for users and another for groups
        '''                                     
        for repo_path in self.__raw_repository_list:
            if repo_path in self.__process_default_right_files.keys():
                yaml_data = self.__process_default_right_files[repo_path]
                if yaml_data is not None:
                    
                    resolve_raw_repository_processes = self.__get_repositories_by_process(
                        repo_path)
                
                    for process in resolve_raw_repository_processes:
                        #fill the lists with the datas
                        if USER_MAIL in yaml_data.keys() and yaml_data[USER_MAIL] is not None:
                            self.__user_default_rights_dict[process] =  yaml_data[USER_MAIL]
                        if GROUP_NAME in yaml_data.keys() and yaml_data[GROUP_NAME] is not None:
                            self.__group_default_rights_dict[process] =  yaml_data[GROUP_NAME]                  

    def __add_python_path_processes(self):
        """
        Build additional process repository base on PYTHONPATH libraries
        The predicate is look for a folder named 'sos_processes' at the library root

        Find for each path the file containing default access rights file for this repository
        """
        
        # check for PYTHONPATH environment variable
        python_path_libraries = environ.get('PYTHONPATH')

        if python_path_libraries is not None and len(python_path_libraries) > 0:

            # Set to list each library of the PYTHONPATH
            libraries = python_path_libraries.split(pathsep)

            for library in libraries:
                processes_modules = [relpath(p, library).replace(sep, '.') for p in Path(
                    library).rglob(f'*/{PROCESSES_MODULE_NAME}/')]

                if processes_modules is not None and len(processes_modules) > 0:
                    self.__raw_repository_list.extend(processes_modules)
                    
                    # From python path, add the automatic default right file if exists
                    file_name = join(library, DEFAULT_RIGHTS_FILE_NAME)
                    if Path(file_name).exists():
                        self.logger.info('--found default right file--')
                        # Read the file
                        # Open and read the yaml file
                        with open(file_name) as stream:
                            yaml_data = yaml.load(stream, Loader=yaml.FullLoader)
                            self.logger.info(f'data from default file:{yaml_data}')
                            if yaml_data is not None:
                                for process_module in processes_modules:
                                    self.__process_default_right_files[process_module] = yaml_data

    def __get_repositories_by_process(self, repository_module_name):
        """ retrieve the list of process name into the specified module name

        :params: repository_module_name, module name (import like name)
        :type: list of strings

        :return: process name list

        """
        # Result process list

        repositories_by_process = {}

        # Convert repository module to corresponding python module in order
        # to retrieve module path
        try:

            # Load module (if not exist a ModuleNotFoundException is loaded
            repository_module = import_module(repository_module_name)

            self.logger.debug(f'Looking for processes into module {repository_module_name}')

            # Get the corresponding filepath
            if repository_module is not None:
                repository_module_path = dirname(repository_module.__file__)

                # Extract all module with SoSProcessFactory.BUILDERS_MODULE_NAME
                # file
                base_id_list = [dirname(relpath(p, repository_module_path)).replace(sep, '.')
                                for p in Path(repository_module_path).rglob(f'*/{BUILDERS_MODULE_NAME}.py')]

                # Manage all process to sort them by processes and
                # repository_module
                for base_process_module in base_id_list:

                    process_full_name = f'{repository_module_name}.{base_process_module}'
                    splitted_process_full_name = process_full_name.split('.')

                    process_name = splitted_process_full_name[-1]
                    process_module = '.'.join(
                        splitted_process_full_name[:-1])

                    if process_module not in repositories_by_process:
                        repositories_by_process[process_module] = []
                    repositories_by_process[process_module].append(
                        process_name)
                    self.logger.debug(f'Find {process_module} / {process_name}')

            else:
                self.logger.warning(
                    f'Unable to load the following module {repository_module_name}')

        except ModuleNotFoundError as error:
            self.logger.critical(
                f'Unable to load the following module {repository_module_name} : {str(error)}')
        except TypeError as error:
            self.logger.critical(
                f'Unable to load the following module {repository_module_name} : {str(error)}')

        return repositories_by_process
