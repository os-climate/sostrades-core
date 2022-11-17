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
    1) Strucrure of Desc_in/Desc_out:
        |_ DESC_IN
            |_ SAMPLING_METHOD (structuring)
                |_ EVAL_INPUTS (namespace: 'ns_doe1', structuring, dynamic : SAMPLING_METHOD == self.DOE_ALGO)             
                |_ SAMPLING_ALGO (structuring, dynamic : SAMPLING_METHOD == self.DOE_ALGO)
                        |_ DESIGN_SPACE (dynamic: SAMPLING_ALGO != None) NB: default DESIGN_SPACE depends on EVAL_INPUTS (Has to be "Not empty")
                        |_ ALGO_OPTIONS (structuring, dynamic: SAMPLING_ALGO != None)
                |_ EVAL_INPUTS_CP (namespace: 'ns_cp', structuring, dynamic : SAMPLING_METHOD == self.CARTESIAN_PRODUCT)
                        |_ GENERATED_SAMPLES(namespace: 'ns_cp', structuring, dynamic: EVAL_INPUTS_CP != None)                 
        |_ DESC_OUT
            |_ SAMPLES_DF
    '''
    VARIABLES = "variable"
    VALUES = "value"
    POSSIBLE_VALUES = 'possible_values'
    TYPE = "type"

    NS_SEP = '.'
    INPUT_TYPE = ['float', 'array', 'int', 'string']

    DESIGN_SPACE = "design_space"
    UPPER_BOUND = "upper_bnd"
    LOWER_BOUND = "lower_bnd"
    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"

    ENABLE_VARIABLE_BOOL = "enable_variable"
    LIST_ACTIVATED_ELEM = "activated_elem"

    N_SAMPLES = "n_samples"
    ALGO = "sampling_algo"
    ALGO_OPTIONS = "algo_options"
    USER_GRAD = 'user'

    # To be defined in the heritage
    is_constraints = None
    INEQ_CONSTRAINTS = 'ineq_constraints'
    EQ_CONSTRAINTS = 'eq_constraints'
    # DESC_I/O
    PARALLEL_OPTIONS = 'parallel_options'

    SAMPLING_METHOD = 'sampling_method'
    DOE_ALGO = 'doe_algo'
    CARTESIAN_PRODUCT = 'cartesian_product'
    available_sampling_methods = [DOE_ALGO, CARTESIAN_PRODUCT]

    EVAL_INPUTS_CP = 'eval_inputs_cp'
    GENERATED_SAMPLES = 'generated_samples'
    SAMPLES_DF = 'samples_df'

    DESC_IN = {'sampling_method': {'type': 'string',
                                   'structuring': True,
                                   'possible_values': available_sampling_methods}}

    DESC_OUT = {
        'samples_df': {'type': 'dataframe',
                       'unit': None,
                       'visibility': SoSWrapp.LOCAL_VISIBILITY}
    }

    def __init__(self, sos_name):
        super().__init__(sos_name)
        self.sampling_method = None
        self.sample_generator_doe = None
        self.sample_generator_cp = None

        # self.previous_algo_name = ""

        self.selected_inputs = None
        self.dict_desactivated_elem = {}
        #self.previous_eval_inputs = None

        self.previous_eval_inputs_cp = None
        self.eval_inputs_cp_filtered = None
        self.eval_inputs_cp_validity = True
        self.samples_gene_df = None

    def setup_sos_disciplines(self, proxy):
        '''
        Overload of setup_sos_disciplines to specify the specific dynamic inputs of sample generator
        '''

        disc_in = proxy.get_data_in()

        if len(disc_in) != 0:
            self.sampling_method = proxy.get_sosdisc_inputs(
                self.SAMPLING_METHOD)
            self.instantiate_sampling_tool()

            # Switch between doe_algo method or cartesian_product method
            if self.sampling_method == self.DOE_ALGO:
                # Reset parameters of the other method to initial values
                # (cleaning)
                self.previous_eval_inputs_cp = None
                self.eval_inputs_cp_filtered = None
                self.eval_inputs_cp_validity = True
                # setup_doe_algo_method
                dynamic_inputs, dynamic_outputs = self.setup_doe_algo_method(
                    proxy)
            elif self.sampling_method == self.CARTESIAN_PRODUCT:
                # Reset parameters of the other method to initial values
                # (cleaning)
                self.selected_inputs = None
                self.dict_desactivated_elem = {}
                self.previous_eval_inputs = None
                # setup_cp_method
                dynamic_inputs, dynamic_outputs = self.setup_cp_method(proxy)
            elif self.sampling_method is not None:
                raise Exception(
                    f"The selected sampling method {self.sampling_method} is not allowed in the sample generator. Please "
                    f"introduce one of the available methods from {self.available_sampling_methods}.")
            else:
                dynamic_inputs = {}
                dynamic_outputs = {}
        else:
            dynamic_inputs = {}
            dynamic_outputs = {}

        proxy.add_inputs(dynamic_inputs)
        proxy.add_outputs(dynamic_outputs)

    def run(self):
        '''
            Overloaded class method
            The generation of samples_df as run time
        '''
        samples_df = None

        if self.sampling_method == self.DOE_ALGO:
            samples_df = self.run_doe()
        elif self.sampling_method == self.CARTESIAN_PRODUCT:
            samples_df = self.run_cp()

        # Loop to raise an error in case the sampling has not been made.
        # If samples' type is dataframe, that means that the previous loop has
        # been entered.
        if isinstance(samples_df, pd.DataFrame):
            pass
        else:
            raise Exception(
                f"Sampling has not been made")

        self.store_sos_outputs_values({'samples_df': samples_df})

    def instantiate_sampling_tool(self):
        """
           Instantiate SampleGenearator once and only if needed
        """
        if self.sampling_method == self.DOE_ALGO:
            if self.sample_generator_doe == None:
                self.sample_generator_doe = DoeSampleGenerator()
        elif self.sampling_method == self.CARTESIAN_PRODUCT:
            if self.sample_generator_cp == None:
                self.sample_generator_cp = CartesianProductSampleGenerator()

    def get_algo_default_options(self, algo_name):
        """
            This algo generate the default options to set for a given doe algorithm
        """

        # In get_options_and_default_values, it is already checked whether the algo_name belongs to the list of possible Gemseo
        # DoE algorithms
        if algo_name in get_available_doe_algorithms():
            algo_options_desc_in, algo_options_descr_dict = self.sample_generator_doe.get_options_and_default_values(
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

    def setup_doe_algo_method(self, proxy):
        """        
        Method that setup the doe_algo method
        """
        dynamic_inputs = {}
        dynamic_outputs = {}

        # Setup dynamic inputs in case of DOE_ALGO selection: i.e. EVAL_INPUTS
        # and SAMPLING_ALGO
        self.setup_dynamic_inputs_for_doe_generator_method(dynamic_inputs)
        # Setup dynamic inputs when SAMPLING_ALGO/EVAL_INPUTS are already set
        self.setup_dynamic_inputs_which_depend_on_sampling_algo(
            proxy, dynamic_inputs)

        return dynamic_inputs, dynamic_outputs

    def setup_dynamic_inputs_for_doe_generator_method(self, dynamic_inputs):
        '''
        Method that setup dynamic inputs in case of DOE_ALGO selection: i.e. EVAL_INPUTS and SAMPLING_ALGO
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated

        '''
        # Get possible values for sampling algorithm name
        available_doe_algorithms = self.sample_generator_doe.get_available_algo_names()
        dynamic_inputs.update({'sampling_algo':
                               {'type': 'string', 'structuring': True,
                                'possible_values': available_doe_algorithms}
                               })
        dynamic_inputs.update({'eval_inputs':
                               {'type': 'dataframe',
                                'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                         'full_name': ('string', None, False)},
                                'dataframe_edition_locked': False,
                                'structuring': True,
                                'visibility': SoSWrapp.SHARED_VISIBILITY,
                                'namespace': 'ns_doe1'}
                               })

    def setup_dynamic_inputs_which_depend_on_sampling_algo(self, proxy, dynamic_inputs):
        """
            Setup dynamic inputs when SAMPLING_ALGO/EVAL_INPUTS are already set
            Manage update of EVAL_INPUTS
            Create or update ALGO_OPTIONS
            Create or update DESIGN_SPACE
        """
        self.setup_algo_options(proxy, dynamic_inputs)
        self.setup_design_space(proxy, dynamic_inputs)

    def setup_algo_options(self, proxy, dynamic_inputs):
        '''
            Method that setup 'algo_options'
            Arguments:
                dynamic_inputs (dict): the dynamic input dict to be updated
        '''
        disc_in = proxy.get_data_in()
        # Dynamic input of algo_options
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

    def setup_design_space(self, proxy, dynamic_inputs):
        '''
        Method that setup 'design_space'
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        '''
        disc_in = proxy.get_data_in()
        # Dynamic input of default design space
        if 'eval_inputs' in disc_in:
            eval_inputs = proxy.get_sosdisc_inputs('eval_inputs')
            if eval_inputs is not None:
                self.selected_inputs = self.reformat_eval_inputs(
                    eval_inputs).tolist()

                default_design_space = pd.DataFrame({'variable': self.selected_inputs,
                                                     'lower_bnd': [None] * len(self.selected_inputs),
                                                     'upper_bnd': [None] * len(self.selected_inputs)
                                                     })

                dynamic_inputs.update({'design_space': {'type': 'dataframe', self.DEFAULT: default_design_space,
                                                        'structuring': True
                                                        }})

    def reformat_eval_inputs(self, eval_inputs):
        '''
        Method that reformat eval_input depending on user's selection

        Arguments:
            eval_inputs (dataframe): 

        Returns:
            eval_inputs_filtered (dataframe) :

        '''
        logic_1 = eval_inputs['selected_input'] == True
        eval_inputs_filtered = eval_inputs[logic_1]
        eval_inputs_filtered = eval_inputs_filtered['full_name']
        return eval_inputs_filtered

    #=========================================================================
    # def setup_generated_samples_for_doe(self, proxy, dynamic_inputs):
    #    '''
    #     Method that setup GENERATED_SAMPLES for doe_algo at configuration time
    #     Arguments:
    #         dynamic_inputs (dict): the dynamic input dict to be updated
    #     Remark: we can implement this method if we want generated_samples for doe at configuration time
    #     '''
    #=========================================================================

    def run_doe(self):
        '''
        Method that generates the desc_out self.SAMPLES_DF from desc_in self.ALGO, self.ALGO_OPTIONS and self.DESIGN_SPACE
        Remark: we could have also generated the GENERATED_SAMPLES at configuration instead of at run
                time if it is a feature that we would like also to have

        Inputs:
            self.GENERATED_SAMPLES (dataframe) : generated samples from sampling method
        Outputs:
            self.SAMPLES_DF (dataframe) : prepared samples for evaluation
        '''

        algo_name = self.get_sosdisc_inputs(self.ALGO)
        algo_options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
        dspace_df = self.get_sosdisc_inputs(self.DESIGN_SPACE)

        design_space = self.create_design_space(
            self.selected_inputs, dspace_df)  # Why self.selected_inputs?
        # self.design_space = design_space

        samples_df = self.sample_generator_doe.generate_samples(
            algo_name, algo_options, design_space)

        return samples_df

    def setup_cp_method(self, proxy):
        '''
        Method that setup the cp method
        '''
        dynamic_inputs = {}
        dynamic_outputs = {}
        # Setup dynamic inputs for CARTESIAN_PRODUCT method: i.e.
        # EVAL_INPUTS_CP
        self.setup_dynamic_inputs_for_cp_generator_method(dynamic_inputs)
        # Setup dynamic inputs which depend on EVAL_INPUTS_CP setting or
        # update: i.e. GENERATED_SAMPLES
        self.setup_dynamic_inputs_which_depend_on_eval_input_cp(
            proxy, dynamic_inputs)

        return dynamic_inputs, dynamic_outputs

    def setup_dynamic_inputs_for_cp_generator_method(self, dynamic_inputs):
        '''
        Method that setup dynamic inputs in case of CARTESIAN_PRODUCT selection: i.e. EVAL_INPUTS_CP

        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated

        '''
        default_in_eval_input_cp = pd.DataFrame({'selected_input': [False],
                                                 'full_name': [''],
                                                 'list_of_values': [[]]})
        dynamic_inputs.update({self.EVAL_INPUTS_CP: {'type': 'dataframe',
                                                     'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                                              'full_name': ('string', None, True),
                                                                              'list_of_values': ('list', None, True)},
                                                     'dataframe_edition_locked': False,
                                                     'structuring': True,
                                                     'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                     'namespace': 'ns_cp',
                                                     'default': default_in_eval_input_cp}})

    def setup_dynamic_inputs_which_depend_on_eval_input_cp(self, proxy, dynamic_inputs):
        '''
        Method that setup dynamic inputs which depend on EVAL_INPUTS_CP setting or update: i.e. GENERATED_SAMPLES
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        '''
        self.eval_inputs_cp_has_changed = False
        disc_in = proxy.get_data_in()
        if self.EVAL_INPUTS_CP in disc_in:
            eval_inputs_cp = proxy.get_sosdisc_inputs(self.EVAL_INPUTS_CP)
            # 1. Manage update status of EVAL_INPUTS_CP
            if not (eval_inputs_cp.equals(self.previous_eval_inputs_cp)):
                self.eval_inputs_cp_has_changed = True
                self.previous_eval_inputs_cp = eval_inputs_cp
            # 2. Manage selection in EVAL_INPUTS_CP
            if eval_inputs_cp is not None:
                # reformat eval_inputs_cp to take into account only useful
                # informations
                self.eval_inputs_cp_filtered = self.reformat_eval_inputs_cp(
                    eval_inputs_cp)
                # Check selected input cp validity
                self.eval_inputs_cp_validity = self.check_eval_inputs_cp(
                    self.eval_inputs_cp_filtered)
                # Setup GENERATED_SAMPLES for cartesian product
                self.setup_generated_samples_for_cp(proxy, dynamic_inputs)

    def setup_generated_samples_for_cp(self, proxy, dynamic_inputs):
        '''
        Method that setup GENERATED_SAMPLES for cartesian product at configuration time
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        '''
        if self.eval_inputs_cp_validity:
            if self.eval_inputs_cp_has_changed:
                dict_of_list_values = self.eval_inputs_cp_filtered.set_index(
                    'full_name').T.to_dict('records')[0]
                self.samples_gene_df = self.sample_generator_cp.generate_samples(
                    dict_of_list_values)
            dynamic_inputs.update({self.GENERATED_SAMPLES: {'type': 'dataframe',
                                                            'dataframe_edition_locked': True,
                                                            'structuring': True,
                                                            'unit': None,
                                                            'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                            'namespace': 'ns_cp',
                                                            'default': self.samples_gene_df}})

        # Set or update GENERATED_SAMPLES in line with selected
        # eval_inputs_cp
        disc_in = proxy.get_data_in()
        if self.GENERATED_SAMPLES in disc_in:
            disc_in[self.GENERATED_SAMPLES]['value'] = self.samples_gene_df

    def reformat_eval_inputs_cp(self, eval_inputs_cp):
        '''
        Method that reformat eval_input_cp depending on user's selection

        Arguments:
            eval_inputs_cp (dataframe): 

        Returns:
            eval_inputs_cp_filtered (dataframe) :

        '''
        logic_1 = eval_inputs_cp['selected_input'] == True
        logic_2 = eval_inputs_cp['list_of_values'].isin([[]])
        logic_3 = eval_inputs_cp['full_name'] is None
        logic_4 = eval_inputs_cp['full_name'] == ''
        eval_inputs_cp_filtered = eval_inputs_cp[logic_1 &
                                                 ~logic_2 & ~logic_3 & ~logic_4]
        eval_inputs_cp_filtered = eval_inputs_cp_filtered[[
            'full_name', 'list_of_values']]
        return eval_inputs_cp_filtered

    def check_eval_inputs_cp(self, eval_inputs_cp_filtered):
        '''
        Method that reformat eval_input_cp depending on user's selection

        Arguments:
            eval_inputs_cp (dataframe): 

        Returns:
            validity (boolean) :

        '''
        is_valid = True
        selected_inputs_cp = list(eval_inputs_cp_filtered['full_name'])
        if len(selected_inputs_cp) < 2:
            LOGGER.warning(
                'Selected_inputs must have at least 2 variables to do a cartesian product')
            is_valid = False
        return is_valid

    def run_cp(self):
        '''
        Method that generates the desc_out self.SAMPLES_DF from desc_in self.GENERATED_SAMPLES
        Here no modification of the samples: but we can imagine that we may remove raws.
        Maybe we do not need a run method here.
        Remark: we could also have generated the GENERATED_SAMPLES at run time instead of at configuration 
                time if it is a feature that we would like also to have

        Inputs:
            self.GENERATED_SAMPLES (dataframe) : generated samples from sampling method
        Outputs:
            self.SAMPLES_DF (dataframe) : prepared samples for evaluation
        '''

        GENERATED_SAMPLES = self.get_sosdisc_inputs(self.GENERATED_SAMPLES)

        samples_df = GENERATED_SAMPLES

        self.store_sos_outputs_values({self.SAMPLES_DF: samples_df})

        return samples_df
