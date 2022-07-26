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

VAR_TYPE_ID = 'type'
VAR_SUBTYPE_ID = 'subtype_descriptor'
VAR_NUMERICAL = 'numerical'
BASE_TYPE_TO_CONVERT = ['dataframe', 'float', 'array']
_NEW_ATTR_TO_SERIALIZE = ['reduced_dm', 'sos_wrapp']


class SoSMDODisciplineException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'


class SoSMDODiscipline(MDODiscipline):
    """**SoSMDODiscipline** is the class which overloads MDODiscipline
    The _run method is overloaded and new methods ( formerly from SoSDiscipline) are added

   """

    def __init__(self, full_name, grammar_type, cache_type, sos_wrapp, reduced_dm):
        '''
        Constructor
        '''
        self.sos_wrapp = sos_wrapp
        self.reduced_dm = reduced_dm
        MDODiscipline.__init__(self, name=full_name,
                               grammar_type=grammar_type,
                               cache_type=cache_type)

    def _run(self):
        self.sos_wrapp.run()

    def check_subtype(self, var_full_name, subtype, type_to_check):
        """This function checks that the subtype given to a list or dictionnary is compliant
        with the defined standard for subtype
        """
        if type(subtype).__name__ != 'dict':
            raise ValueError(
                f' subtype of variable {var_full_name} must be a dictionnary')
        elif list(subtype.keys())[0] != type_to_check or len(list(subtype.keys())) != 1:
            raise ValueError(
                f' subtype of variable {var_full_name} should have as unique key the keyword {type_to_check}')
        elif type(subtype[type_to_check]).__name__ != 'dict':
            if subtype[type_to_check] == type_to_check:
                raise ValueError(
                    f' subtype of variable {var_full_name} should indicate the type inside the {type_to_check}')
            else:
                return subtype[type_to_check]
        else:

            return self.check_subtype(var_full_name, subtype[type_to_check],
                                      list(subtype[type_to_check].keys())[0])

    def filter_variables_to_convert(self, list_to_filter, write_logs=False):
        """  filter variables to convert
        """
        filtered_keys = []

        for variable in list_to_filter:
            will_be_converted = False
            variable_local_data = self.reduced_dm[variable]
            is_numerical = variable_local_data[VAR_NUMERICAL]
            if not is_numerical:
                type = variable_local_data[VAR_TYPE_ID]
                if type in BASE_TYPE_TO_CONVERT:
                    filtered_keys.append(variable)
                    will_be_converted = True
                elif type not in ['string', 'string_list', 'string_list_list', 'int_list', 'float_list', 'bool',
                                  'dict_list', 'df_dict']:
                    subtype = variable_local_data.get(VAR_SUBTYPE_ID)
                    if subtype is not None:
                        final_type = SoSMDODiscipline.check_subtype(
                            variable, subtype, type)
                        if final_type in BASE_TYPE_TO_CONVERT:
                            filtered_keys.append(variable)
                            will_be_converted = True

            if not will_be_converted and write_logs:
                self.LOGGER.info(
                    f'variable {variable} in strong couplings wont be taken into consideration in residual computation')
        return filtered_keys

    def get_input_data_names(self, filtered_inputs=False):  # type: (...) -> List[str]
        """Return the names of the input variables.

        Returns:
            The names of the input variables.
        """
        if not filtered_inputs:
            return self.input_grammar.get_data_names()
        else:
            return self.filter_variables_to_convert(self.input_grammar.get_data_names())

    def get_output_data_names(self, filtered_outputs=False):  # type: (...) -> List[str]
        """Return the names of the output variables.

        Returns:
            The names of the output variables.
        """
        if not filtered_outputs:
            return self.output_grammar.get_data_names()
        else:
            return self.filter_variables_to_convert(self.output_grammar.get_data_names())

    def get_attributes_to_serialize(self):  # pylint: disable=R0201
        """Define the names of the attributes to be serialized.

        Shall be overloaded by disciplines

        Returns:
            The names of the attributes to be serialized.
        """
        # pylint warning ==> method could be a function but when overridden,
        # it is a function==> self is required

        return super().get_attributes_to_serialize() + _NEW_ATTR_TO_SERIALIZE
