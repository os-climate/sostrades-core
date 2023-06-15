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
"""
from json import dumps
class DataManagementDiscipline :
    """
    Class to store discipline data
    """
    def __init__(self):

        # model of the discipline
        self.model_name_full_path = ''

        # name of the discipline
        self.discipline_label = '' 

        # Namespace associated to the discipline
        self.namespace = ''

        # Discipline maturity (determined using the discipline maturity)
        self.maturity = ''

        #inputs of the discipline
        self.disciplinary_inputs = {}

        #outputs of the discipline
        self.disciplinary_outputs = {}

        #inputs of the discipline
        self.numerical_parameters = {}

    def to_json(self):
        return self.to_dict()

    def to_dict(self):
        dict_obj = {}
        # Serialize name attribute
        dict_obj.update({'model_name_full_path': self.model_name_full_path})

        # Serialize data attribute
        dict_obj.update({'discipline_label': self.discipline_label})

        # Serialize node_type attribute
        dict_obj.update({'namespace': self.namespace})

        # Serialize status attribute
        dict_obj.update({'maturity': self.maturity})

        # Serialize full_namespace attribute
        dict_obj.update({'disciplinary_inputs': self.disciplinary_inputs})

        # Serialize identifier attribute
        dict_obj.update({'disciplinary_outputs': self.disciplinary_outputs})

        # Serialize maturity attribute
        dict_obj.update({'numerical_parameters': self.numerical_parameters})

        return dict_obj
