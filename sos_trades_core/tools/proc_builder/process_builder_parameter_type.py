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


class ProcessBuilderParameterType:

    # Define some constant regarding structure to manage
    PROCESS_NAME = 'process_name'
    USECASE_INFO = 'usecase_info'
    USECASE_NAME = 'usecase_name'
    USECASE_TYPE = 'usecase_type'
    USECASE_IDENTIFIER = 'usecase_identifier'
    USECASE_DATA = 'usecase_data'
    PROCESS_REPOSITORY = 'process_repository'

    def __init__(self, process_name='', repository_name='', usecase_name='',
                 usecase_type='', usecase_identifier=-1, usecase_data={}):
        '''
        constructor
        :param process_name: name of the process
        :type process_name: str
        :param repository_name:repository name of the process
        :type repository_name: str
        :param usecase_name: name of the selected data source
        :type usecase_name: str
        :param usecase_type: type of the selected data source
        :type usecase_type: str
        :param usecase_identifier: data source identifier
        :type usecase_identifier: int
        :param usecase_data: data source data
        :type usecase_data: dict
        '''
        self.__process_name = process_name
        self.__process_repository = repository_name
        self.__usecase_name = usecase_name
        self.__usecase_type = usecase_type
        self.__usecase_identifier = usecase_identifier
        self.__usecase_data = usecase_data

    @property
    def process_name(self):
        return self.__process_name

    @process_name.setter
    def process_name(self, value):
        self.__process_name = value

    @property
    def process_repository(self):
        return self.__process_repository

    @process_repository.setter
    def process_repository(self, value):
        self.__process_repository = value

    @property
    def usecase_name(self):
        return self.__usecase_name

    @usecase_name.setter
    def usecase_name(self, value):
        self.__usecase_name = value

    @property
    def usecase_type(self):
        return self.__usecase_type

    @usecase_type.setter
    def usecase_type(self, value):
        self.__usecase_type = value

    @property
    def usecase_identifier(self):
        return self.__usecase_identifier

    @usecase_identifier.setter
    def usecase_identifier(self, value):
        self.__usecase_identifier = value

    @property
    def usecase_data(self):
        return self.__usecase_data

    @usecase_data.setter
    def usecase_data(self, usecase_data):
        self.__usecase_data = usecase_data

    @property
    def has_usecase(self):
        return self.usecase_name != ''

    @property
    def has_valid_study_identifier(self):
        return self.usecase_type == 'Study' and self.usecase_identifier > 0

    def to_data_manager_dict(self):
        '''
        Convert current instance to data manager value dictionary
        :return: dict
        '''

        result = {}
        result[ProcessBuilderParameterType.PROCESS_REPOSITORY] = self.process_repository
        result[ProcessBuilderParameterType.PROCESS_NAME] = self.process_name
        result[ProcessBuilderParameterType.USECASE_INFO] = {}
        result[ProcessBuilderParameterType.USECASE_INFO][ProcessBuilderParameterType.USECASE_NAME] = self.usecase_name
        result[ProcessBuilderParameterType.USECASE_INFO][ProcessBuilderParameterType.USECASE_TYPE] = self.usecase_type
        result[ProcessBuilderParameterType.USECASE_INFO][ProcessBuilderParameterType.USECASE_IDENTIFIER] = self.usecase_identifier
        result[ProcessBuilderParameterType.USECASE_DATA] = self.usecase_data

        return result

    @staticmethod
    def create(data: dict):
        '''
        Static method to build ProcBuilderModalType instance using dictionary from data manager parameter value
        :param data: dictionary of values
        :type data: dict
        :return: ProcBuilderModalType instance
        '''
        repository_name = ''
        process_name = ''
        usecase_name = ''
        usecase_type = ''
        usecase_identifier = -1
        usecase_data = {}

        if data is not None:
            if ProcessBuilderParameterType.PROCESS_REPOSITORY in data:
                repository_name = data[ProcessBuilderParameterType.PROCESS_REPOSITORY]

            if ProcessBuilderParameterType.PROCESS_NAME in data:
                process_name = data[ProcessBuilderParameterType.PROCESS_NAME]

            if ProcessBuilderParameterType.USECASE_INFO in data:
                if ProcessBuilderParameterType.USECASE_NAME in data[ProcessBuilderParameterType.USECASE_INFO]:
                    usecase_name = data[ProcessBuilderParameterType.USECASE_INFO][ProcessBuilderParameterType.USECASE_NAME]

                if ProcessBuilderParameterType.USECASE_TYPE in data[ProcessBuilderParameterType.USECASE_INFO]:
                    usecase_type = data[ProcessBuilderParameterType.USECASE_INFO][ProcessBuilderParameterType.USECASE_TYPE]

                if ProcessBuilderParameterType.USECASE_IDENTIFIER in data[ProcessBuilderParameterType.USECASE_INFO]:
                    usecase_identifier = data[ProcessBuilderParameterType.USECASE_INFO][ProcessBuilderParameterType.USECASE_IDENTIFIER]

            if ProcessBuilderParameterType.USECASE_DATA in data:
                usecase_data = data[ProcessBuilderParameterType.USECASE_DATA]

        return ProcessBuilderParameterType(process_name, repository_name, usecase_name,
                                    usecase_type, usecase_identifier, usecase_data)