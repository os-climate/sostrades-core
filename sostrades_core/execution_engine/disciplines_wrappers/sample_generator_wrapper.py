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
import copy
import re

import platform
from tqdm import tqdm
import time

from numpy import array, ndarray, delete, NaN

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.execution_engine.disciplines_wrappers.eval_wrapper import EvalWrapper

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import DoeSampleGenerator
from sostrades_core.execution_engine.sample_generators.cartesian_product_sample_generator import CartesianProductSampleGenerator
import pandas as pd
from collections import ChainMap
from gemseo.api import get_available_doe_algorithms

# get module logger not sos logger
import logging
LOGGER = logging.getLogger(__name__)


class SampleGeneratorWrapper(SoSWrapp):
    '''
    Generic SampleGenerator class
    '''

    # Design space dataframe headers
    VARIABLES = "variable"
    VALUES = "value"
    UPPER_BOUND = "upper_bnd"
    LOWER_BOUND = "lower_bnd"
    TYPE = "type"
    ENABLE_VARIABLE_BOOL = "enable_variable"
    LIST_ACTIVATED_ELEM = "activated_elem"
    POSSIBLE_VALUES = 'possible_values'
    N_SAMPLES = "n_samples"
    DESIGN_SPACE = "design_space"

    ALGO = "sampling_algo"
    ALGO_OPTIONS = "algo_options"
    USER_GRAD = 'user'

    # To be defined in the heritage
    is_constraints = None
    INEQ_CONSTRAINTS = 'ineq_constraints'
    EQ_CONSTRAINTS = 'eq_constraints'
    # DESC_I/O
    PARALLEL_OPTIONS = 'parallel_options'

    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    NS_SEP = '.'
    INPUT_TYPE = ['float', 'array', 'int', 'string']

    SAMPLING_METHOD = 'sampling_method'
    DOE_GENERATOR = 'DOE_GENERATOR'
    CARTESIAN_PRODUCT_GENERATOR = 'CARTESIAN_PRODUCT_GENERATOR'

    available_sampling_methods = [DOE_GENERATOR, CARTESIAN_PRODUCT_GENERATOR]
    DESC_IN = {'sampling_method': {'type': 'string', 'structuring': True, 'possible_values': available_sampling_methods}}

    DESC_OUT = {
        'samples_df': {'type': 'dataframe', 'unit': None, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                       'namespace': 'ns_doe1'}
    }

    default_algo_options = {
        'n_samples': 'default',
        'alpha': 'orthogonal',
        'eval_jac': False,
        'face': 'faced',
        'iterations': 5,
        'max_time': 0,
        'seed': 1,
        'center_bb': 'default',
        'center_cc': 'default',
        'criterion': 'default',
        'levels': 'default'
    }


    def __init__(self, sos_name):
        super().__init__(sos_name)
        self.sample_generator = None
        self.sampling_method = None
        # self.previous_algo_name = ""
        self.previous_sampling_method = ""
        self.selected_inputs = None
        self.eval_in_list = None
        self.selected_inputs = None
        self.dict_desactivated_elem = {}

        # self.instantiate_sampling_tool()

    def instantiate_sampling_tool(self):

        if self.sampling_method == self.DOE_GENERATOR:
            self.sample_generator = DoeSampleGenerator()
        elif self.sampling_method == self.CARTESIAN_PRODUCT_GENERATOR:
            self.sample_generator = CartesianProductSampleGenerator()
        else:
            raise Exception(
                f"The selected sampling method {self.sampling_method} is not allowed in the sample generator. Please "
                f"introduce one of the available methods from {self.available_sampling_methods}.")


    def get_algo_default_options(self, algo_name):
        """This algo generate the default options to set for a given doe algorithm
        """

        # In get_options_and_default_values, it is already checked whether the algo_name belongs to the list of possible Gemseo
        # DoE algorithms
        if algo_name in get_available_doe_algorithms():
            algo_options_desc_in, algo_options_descr_dict = self.sample_generator.get_options_and_default_values(
                algo_name)
            return algo_options_desc_in
        else:
            raise Exception(
                f"A DoE algorithm which is not available in GEMSEO has been selected.")

    def create_design_space(self, selected_inputs, dspace_df):
        """
        create_design_space with variables names based on selected_inputs (if dspace_df is not None)

        Arguments:
            selected_inputs (list): list of selected variables (the true variables in eval_inputs Desc_in)
            dspace_df (dataframe): design space in Desc_in format     

        Returns:
             design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs
        """

        design_space = None
        if dspace_df is not None:
            dspace_df_updated = self.update_design_space(
                selected_inputs, dspace_df)
            design_space = self.create_gemseo_dspace_from_dspace_df(
                dspace_df_updated)
        return design_space

    def update_design_space(self, selected_inputs, dspace_df):
        """
        update dspace_df (design space in Desc_in format)   

        Arguments:
            selected_inputs (list): list of selected variables (the true variables in eval_inputs Desc_in)
            dspace_df (dataframe): design space in Desc_in format     

        Returns:
             dspace_df_updated (dataframe): updated dspace_df        

        """
        lower_bounds = dspace_df[self.LOWER_BOUND].tolist()
        upper_bounds = dspace_df[self.UPPER_BOUND].tolist()
        values = lower_bounds
        enable_variables = [True for _ in selected_inputs]
        dspace_df_updated = pd.DataFrame({self.VARIABLES: selected_inputs,
                                          self.VALUES: values,
                                          self.LOWER_BOUND: lower_bounds,
                                          self.UPPER_BOUND: upper_bounds,
                                          self.ENABLE_VARIABLE_BOOL: enable_variables,
                                          self.LIST_ACTIVATED_ELEM: [[True]] * len(selected_inputs)})
        # TODO: Hardcoded as in EEV3, but not differenciating between array or
        # not.
        return dspace_df_updated

    def create_gemseo_dspace_from_dspace_df(self, dspace_df):
        """
        Create gemseo dspace from sostrades updated dspace_df 
        It parses the dspace_df DataFrame to create the gemseo DesignSpace

        Arguments:
            dspace_df (dataframe): updated dspace_df     

        Returns:
            design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs
        """
        names = list(dspace_df[self.VARIABLES])
        values = list(dspace_df[self.VALUES])
        l_bounds = list(dspace_df[self.LOWER_BOUND])
        u_bounds = list(dspace_df[self.UPPER_BOUND])
        enabled_variable = list(dspace_df[self.ENABLE_VARIABLE_BOOL])
        list_activated_elem = list(dspace_df[self.LIST_ACTIVATED_ELEM])
        design_space = DesignSpace()
        for dv, val, lb, ub, l_activated, enable_var in zip(names, values, l_bounds, u_bounds, list_activated_elem,
                                                            enabled_variable):

            # check if variable is enabled to add it or not in the design var
            if enable_var:

                self.dict_desactivated_elem[dv] = {}

                if [type(val), type(lb), type(ub)] == [str] * 3:
                    val = val
                    lb = lb
                    ub = ub
                name = dv
                if type(val) != list and type(val) != ndarray:
                    size = 1
                    var_type = ['float']
                    l_b = array([lb])
                    u_b = array([ub])
                    value = array([val])
                else:
                    # check if there is any False in l_activated
                    if not all(l_activated):
                        index_false = l_activated.index(False)
                        self.dict_desactivated_elem[dv] = {
                            'value': val[index_false], 'position': index_false}

                        val = delete(val, index_false)
                        lb = delete(lb, index_false)
                        ub = delete(ub, index_false)

                    size = len(val)
                    var_type = ['float'] * size
                    l_b = array(lb)
                    u_b = array(ub)
                    value = array(val)
                design_space.add_variable(
                    name, size, var_type, l_b, u_b, value)
        return design_space

    def run(self):

        samples = None

        if self.sampling_method == self.DOE_GENERATOR:
            algo_name = self.get_sosdisc_inputs(self.ALGO)
            algo_options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
            dspace_df = self.get_sosdisc_inputs(self.DESIGN_SPACE)

            design_space = self.create_design_space(
                self.selected_inputs, dspace_df)
            # self.design_space = design_space

            samples = self.sample_generator.generate_samples(
                algo_name, algo_options, design_space)
        elif self.sampling_method == self.CARTESIAN_PRODUCT:
            pass
        else:
            # FIXME: this case should not arrive since exception would already had been raised during intanstiation of sampling tool. Else in case more tools are introduced.
            raise Exception(
                f"The selected sampling method {self.sampling_method} is not allowed in the sample generator. Please "
                f"introduce one of the available methods from {self.available_sampling_methods}.")

        # Loop to raise an error in case the sampling has not been made.
        # If samples' type is dataframe, that means that the previous loop has been entered.
        if isinstance(samples, pd.DataFrame):
            pass
        else:
            raise Exception(
                f"Sampling has not been made")

        self.store_sos_outputs_values({'samples_df': samples})

    def setup_sos_disciplines(self, proxy):

        dynamic_inputs = {}
        dynamic_outputs = {}
        disc_in = proxy.get_data_in()

        if len(disc_in) != 0:

            self.sampling_method = proxy.get_sosdisc_inputs(self.SAMPLING_METHOD)
            self.instantiate_sampling_tool()

            # sampling_method_has_changed = False
            # if self.previous_sampling_method != self.sampling_method:
            #     sampling_method_has_changed = True
            #     self.previous_sampling_method = self.sampling_method

            if self.sampling_method == self.DOE_GENERATOR:

                dynamic_inputs, dynamic_outputs = self.update_dynamic_inputs_outputs_doe(proxy,disc_in)
            elif self.sampling_method == self.CARTESIAN_PRODUCT_GENERATOR:
                pass
            else:
                #FIXME: this case should not arrive since exception would already had been raised during intanstiation of sampling tool. Else in case more tools are introduced.
                raise Exception(
                    f"The selected sampling method {self.sampling_method} is not allowed in the sample generator. Please "
                    f"introduce one of the available methods from {self.available_sampling_methods}.")

        proxy.add_inputs(dynamic_inputs)
        proxy.add_outputs(dynamic_outputs)

    def update_dynamic_inputs_outputs_doe(self, proxy, disc_in):

        dynamic_inputs = {}
        dynamic_outputs = {}
        # selected_inputs_has_changed = False

        # Dynamic input of old 'nominal' DESC_IN of DoEWrapper
        # Get possible values for sampling algorithm name
        available_doe_algorithms = self.sample_generator.get_available_algo_names()
        dynamic_inputs.update({'sampling_algo':
                                   {'type': 'string', 'structuring': True, 'possible_values': available_doe_algorithms},
                               'eval_inputs':
                                   {'type': 'dataframe',
                                    'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                             'full_name': ('string', None, False)},
                                    'dataframe_edition_locked': False,
                                    'structuring': True,
                                    'visibility': SoSWrapp.SHARED_VISIBILITY,
                                    'namespace': 'ns_doe1'},
                               })

        # algo_name = proxy.get_sosdisc_inputs(self.ALGO)
        # eval_inputs = proxy.get_sosdisc_inputs('eval_inputs')

        # Dynamic input of default algo options
        if self.ALGO in disc_in:
            algo_name = proxy.get_sosdisc_inputs(self.ALGO)

            # algo_name_has_changed = False
            # if self.previous_algo_name != algo_name:
            #     algo_name_has_changed = True
            #     self.previous_algo_name = algo_name

            if algo_name is not None:  # and algo_name_has_changed:
                default_dict = self.get_algo_default_options(algo_name)
                dynamic_inputs.update({'algo_options': {'type': 'dict', self.DEFAULT: default_dict,
                                                        'dataframe_edition_locked': False,
                                                        'structuring': True,

                                                        'dataframe_descriptor': {
                                                            self.VARIABLES: ('string', None, False),
                                                            self.VALUES: ('string', None, True)}}})
                all_options = list(default_dict.keys())
                # if 'algo_options' in disc_in and algo_name_has_changed:
                #     disc_in['algo_options']['value'] = default_dict
                if 'algo_options' in disc_in and disc_in['algo_options']['value'] is not None and list(
                        disc_in['algo_options']['value'].keys()) != all_options:
                    options_map = ChainMap(
                        disc_in['algo_options']['value'], default_dict)
                    disc_in['algo_options']['value'] = {
                        key: options_map[key] for key in all_options}

        # Dynamic input of default design space
        if 'eval_inputs' in disc_in:
            eval_inputs = proxy.get_sosdisc_inputs('eval_inputs')
            if eval_inputs is not None:
                self.selected_inputs = eval_inputs[eval_inputs['selected_input']
                                                   == True]['full_name'].tolist()

                default_design_space = pd.DataFrame({'variable': self.selected_inputs,

                                                     # FIXME: Ask about this
                                                     # default setup
                                                     'lower_bnd': [None] * len(self.selected_inputs),
                                                     # [[0.0, 0.0] if proxy.ee.dm.get_data(var,
                                                     #                                              'type') == 'array' else 0.0
                                                     # for var in
                                                     # self.eval_in_list],
                                                     # FIXME: Ask about this
                                                     # default setup
                                                     'upper_bnd': [None] * len(self.selected_inputs)
                                                     # [[10.0, 10.0] if proxy.ee.dm.get_data(var,
                                                     #                                                'type') == 'array' else 10.0
                                                     # for var in
                                                     # self.eval_in_list]
                                                     })

                dynamic_inputs.update({'design_space': {'type': 'dataframe', self.DEFAULT: default_design_space,
                                                        'structuring': True
                                                        }})
                # if 'design_space' in disc_in and selected_inputs_has_changed:
                #     disc_in['design_space']['value'] = default_design_space

        return dynamic_inputs, dynamic_outputs