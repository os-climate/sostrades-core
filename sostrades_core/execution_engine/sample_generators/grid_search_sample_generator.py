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
from builtins import NotImplementedError

from sostrades_core.execution_engine.sample_generators.cartesian_product_sample_generator import CartesianProductSampleGenerator
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import DoeSampleGenerator, DoeSampleTypeError

import pandas as pd
import numpy as np

import itertools

import logging
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
        # FIXME: currently it is a doesamplegenerator using cartesianproductsamplegenerator in composition, design might be improved
        self.cp_generator = CartesianProductSampleGenerator(logger=logger)
        self.retreive_cp_generator_attributes() # TODO: improve design

    def setup(self, proxy):
        dynamic_inputs, dynamic_outputs = super().setup(proxy)
        self.setup_gs(dynamic_inputs, proxy)
        return dynamic_inputs, dynamic_outputs

    def generate_samples(self, *args, **kwargs):
        return self.cp_generator.generate_samples(*args, **kwargs)

    def get_arguments(self, wrapper):
        return self.cp_generator.get_arguments(wrapper)

    def setup_gs(self, dynamic_inputs, proxy):
        """
        Method that setup dynamic inputs which depend on EVAL_INPUTS_CP setting or update: i.e. GENERATED_SAMPLES
        with specificities for the GridSearch sampling method.
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        """
        disc_in = proxy.get_data_in()
        self.cp_generator.eval_inputs_cp_has_changed = False
        if proxy.DESIGN_SPACE in disc_in:
            eval_inputs = proxy.get_sosdisc_inputs(proxy.EVAL_INPUTS)
            design_space = proxy.get_sosdisc_inputs(proxy.DESIGN_SPACE)
            # link doe-like inputs to cp attributes in the framework of GridSearch
            eval_inputs_cp = self.get_eval_inputs_cp_for_gs(eval_inputs, design_space)
            self.cp_generator.setup_eval_inputs_cp_and_generated_samples(dynamic_inputs, eval_inputs_cp, proxy)
            self.retreive_cp_generator_attributes()

    # def setup_generated_sample(self, dynamic_inputs, proxy):
    #     self.cp_generator.setup_generated_sample(dynamic_inputs, proxy)

    def retreive_cp_generator_attributes(self): # TODO: improve design
        self.previous_eval_inputs_cp = self.cp_generator.previous_eval_inputs_cp
        self.eval_inputs_cp_has_changed = self.cp_generator.eval_inputs_cp_has_changed
        self.eval_inputs_cp_filtered = self.cp_generator.eval_inputs_cp_filtered
        self.eval_inputs_cp_validity = self.cp_generator.eval_inputs_cp_validity

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
        return self.cp_generator.is_ready_to_sample(proxy)
