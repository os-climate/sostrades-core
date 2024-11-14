'''
Copyright 2024 Capgemini
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

import sys


def compute_data_size_in_Mo(data_value) -> float:
        '''
        Compute the size of an object and convert it in Mo
        :param data_value: value of the data to be checked
        :type data_value: depends of the type of the variable
        :return: the size in Mo (float)
        '''
        data_size = 0
        if data_value is not None:

            # test deep size of the object
            data_size = sys.getsizeof(data_value)
        return data_size / 1024 / 1024
