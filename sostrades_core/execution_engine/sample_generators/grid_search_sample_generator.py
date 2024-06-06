'''
Copyright 2023 Capgemini

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

import logging

import numpy as np

from sostrades_core.execution_engine.sample_generators.cartesian_product_sample_generator import (
    CartesianProductSampleGenerator,
)
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import (
    DoeSampleGenerator,
    DoeSampleTypeError,
)

LOGGER = logging.getLogger(__name__)

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''


class GridSearchSampleGeneratorTypeError(DoeSampleTypeError):
    pass


class GridSearchSampleGenerator(DoeSampleGenerator):
    '''
    Caresian Product class that generates sampling
    '''
    GENERATOR_NAME = "GRID_SEARCH_GENERATOR"

    def __init__(self, logger: logging.Logger):
        '''
        Constructor
        '''
        super().__init__(logger=logger)
        # TODO: currently it is a doesamplegenerator using cartesianproductsamplegenerator in composition,
        #  design might be improved as only need to setup a design space so it could be detached from DoE setup
        self.cp_generator = CartesianProductSampleGenerator(logger=logger)

    def setup(self, proxy):
        dynamic_inputs, dynamic_outputs = super().setup(proxy)
        return dynamic_inputs, dynamic_outputs

    def generate_samples(self, *args, **kwargs):
        return self.cp_generator.generate_samples(*args, **kwargs)

    def get_arguments(self, wrapper):
        design_space = wrapper.get_sosdisc_inputs(wrapper.DESIGN_SPACE)
        eval_inputs = wrapper.get_sosdisc_inputs(wrapper.EVAL_INPUTS)
        eval_inputs_cp = self.get_eval_inputs_cp_for_gs(eval_inputs, design_space)
        dict_of_list_values = self.cp_generator.filter_eval_inputs_cp(eval_inputs_cp, wrapper)
        return [], {"dict_of_list_values": dict_of_list_values}

    def get_eval_inputs_cp_for_gs(self, eval_inputs, design_space): # FIXME: use class variables
        """
        Method that modifies Doe-type eval_inputs into eval_inputs_cp to use CartesianProduct for GridSearch.

        Arguments:
            eval_inputs(dataframe): Doe-like eval_inputs.
            design_space(dataframe): GridSearch design space with nb_points.
        Returns:
            eval_inputs_cp(dataframe): with extra column with the values for CartesianProduct SampleGenerator.
        """
        if eval_inputs is not None and design_space is not None:
            lists_of_values = []
            for idx, var_row in eval_inputs.iterrows():
                if var_row['selected_input'] is True and var_row['full_name'] in design_space['variable'].tolist():
                    dspace_row = design_space[design_space['variable'] == var_row['full_name']].iloc[0]
                    lb = dspace_row['lower_bnd']
                    ub = dspace_row['upper_bnd']
                    nb_points = dspace_row['nb_points']
                    lists_of_values.append(np.linspace(lb, ub, nb_points).tolist())
                else:
                    lists_of_values.append([])

            eval_inputs_cp = eval_inputs.assign(list_of_values=lists_of_values)
            return eval_inputs_cp

    def is_ready_to_sample(self, proxy):
        disc_in = proxy.get_data_in()
        if proxy.DESIGN_SPACE in disc_in:
            # avoid empty sampling
            _args, _kwargs = self.get_arguments(proxy)
            return bool(_kwargs["dict_of_list_values"])
        else:
            # otherwise it is intermediate config. stage
            return False
