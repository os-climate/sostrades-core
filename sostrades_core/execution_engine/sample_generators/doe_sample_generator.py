'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/17-2023/11/03 Copyright 2023 Capgemini

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
from typing import Optional
import logging

from sostrades_core.execution_engine.sample_generators.abstract_sample_generator import AbstractSampleGenerator,\
    SampleTypeError
from sostrades_core.tools.design_space import design_space as dspace_tool
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.api import get_available_doe_algorithms
from gemseo.api import get_algorithm_options_schema
from gemseo.api import compute_doe
from numpy import array, ndarray, delete, NaN

from gemseo.algos.doe.doe_factory import DOEFactory

import pandas as pd
from gemseo.utils.source_parsing import get_options_doc

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from collections import ChainMap


class DoeSampleTypeError(SampleTypeError):
    pass


class DoeSampleGenerator(AbstractSampleGenerator):
    '''
    Abstract class that generates sampling
    '''
    GENERATOR_NAME = "DOE_GENERATOR"
    VARIABLES = dspace_tool.VARIABLES
    VALUES = dspace_tool.VALUES
    UPPER_BOUND = dspace_tool.UPPER_BOUND
    LOWER_BOUND = dspace_tool.LOWER_BOUND
    ENABLE_VARIABLE_BOOL = dspace_tool.ENABLE_VARIABLE_BOOL
    LIST_ACTIVATED_ELEM = dspace_tool.LIST_ACTIVATED_ELEM
    NB_POINTS = "nb_points"

    
    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    N_PROCESSES = 'n_processes'
    WAIT_TIME_BETWEEN_SAMPLES = 'wait_time_between_samples'

    N_SAMPLES = "n_samples"

    # We remove 'CustomDOE' from gemseo
    # https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html#customdoe

    # We remove 'DiagonalDOE' from gemseo gemseo.algos.doe.lib_scalable
    # https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html#diagonaldoe
    # This algorithm is special: it has parameters "reverse" that can have name of variable
    # Do we want it in SoSTrades. Does it works also or not ?
    UNSUPPORTED_GEMSEO_ALGORITHMS = ['CustomDOE', 'DiagonalDOE']
    # algorithms not listed below will have input constrained to floats and arrays
    TYPE_PERMISSIVE_ALGORITHMS = {'fullfact', 'OT_FULLFACT', 'pbdesign', 'ff2n'}

    def __init__(self, logger:Optional[logging.Logger]=None):
        '''
        Constructor
        '''
        logger_aux = logger
        if logger_aux is None:
            logger_aux = logging.getLogger(__name__)
        #- inits super class
        super().__init__(self.GENERATOR_NAME, logger=logger_aux)
        #- create attributes
        self.doe_factory = None
        self.__available_algo_names = None
        #- set attribute values
        self._reload()

        self.selected_inputs = []
        self.selected_inputs_types = {}
        # self.dict_desactivated_elem = {}
        
    def _reload(self):
        '''
        Reloads all attributes of the class
        - creates the DOEFactory
        - creates the DOEFactory from GEMSEO
        '''
        # DOEFactory is instantiated once here
        self.doe_factory = DOEFactory()
        # all the DOE algorithms in GEMSEO that are available in current environment
        all_names = self.doe_factory.algorithms
        # filter with the unsupported GEMSEO algorithms.
        self.__available_algo_names = list(set(all_names)-set(self.UNSUPPORTED_GEMSEO_ALGORITHMS))
        
    def get_available_algo_names(self):
        '''
        Method that provides the already created list of available algo_names, filtered by the unsupported ones

        Returns:
             algo_names_list (list): list of available algo names
        '''
        return self.__available_algo_names
    
    def _check_algo_name(self, algo_name):
        '''
        Check provided algo name before getting its algo_options
        Arguments:
            algo_name (string): name of the numerical algorithm
        Raises:
            Exception if sampling_algo_name is not in the list of available algorithms
        '''
        algo_names_list = self.get_available_algo_names()
        if algo_name not in algo_names_list:
            if algo_name in self.UNSUPPORTED_GEMSEO_ALGORITHMS:
                raise Exception(
                    f"The provided algorithm name {algo_name} is not allowed in doe sample generator")
            else:
                raise Exception(
                    f"The provided algorithm name {algo_name} is not in the available algorithm list : {algo_names_list}")

    def get_options_and_default_values(self, sampling_algo_name):
        '''
        Method that provides the list of options of an algorithm with there default values (if any) and description

        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
        Returns:
            the Sample Generator expected inputs (as DESC_IN format)
                                (to be provided to proxy i/o grammars)
            More precisely:
             algo_options (dict): dict of algo options with default values (if any). It is in algo_options desci_in format
             algo_options_descr_dict (dict): dict of description of algo options
             e.g. https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html#fullfact

        '''
        # check sampling_algo_name
        self._check_algo_name(sampling_algo_name)

        # create the algo library
        algo_lib = self.doe_factory.create(sampling_algo_name)

        # Remark: The following lines of code should be in gemseo
        # We should use only one line or two provided by gemseo
#         default_opt = algo_lib._get_options()
#        # provided options and not providing default options

        fn = algo_lib.__class__._get_options

        # retrieve description of options provided in docstrings
        algo_options_descr_dict = get_options_doc(fn)
        
        # retrieve default algo options and values
        def get_default_args(func):
            import inspect
            signature = inspect.signature(func)
            return {
                k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty
            }
        algo_options = get_default_args(fn)
        
        # get option keys dedicated to algo
        opts_gram = algo_lib.init_options_grammar(sampling_algo_name)
        opt_to_keep = opts_gram.get_data_names()
        
        # remove options that are not in the grammar of the algo
        all_options = list(algo_options.keys())
        for k in all_options:
            if k not in opt_to_keep:
                algo_options.pop(k,None)
                algo_options_descr_dict.pop(k,None)

        return algo_options, algo_options_descr_dict

    def _check_options(self, sampling_algo_name, algo_options, design_space):
        '''
        Check provided options before sample generation
        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm        
        '''

        if self.N_SAMPLES not in algo_options:
            self.logger.warning("N_samples is not defined; pay attention you use fullfact algo "
                           "and that levels are well defined")

        pass

    def _check_samples(self, samples_df):
        '''
        Method that checks the sample output type
        Arguments:
            samples_df (dataframe) : generated samples 
        Raises:
            Exception if samples_df is not a dataframe                   
        '''
        if not(isinstance(samples_df, pd.DataFrame)):
            msg = "Expected sampling output type should be pandas.core.frame.DataFrame"
            msg += "however sampling type of sampling generator <%s> " % str(
                self.__class__.__name__)
            msg += "is <%s> " % str(type(samples_df))
            raise SampleTypeError()

    def _generate_samples(self, sampling_algo_name, algo_options, design_space):
        '''
        Method that generate samples

        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm
            design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs

        Returns:
            samples_df (dataframe) : generated samples
        '''
        normalized_samples = self._generate_normalized_samples(
            sampling_algo_name, algo_options, design_space)
        unnormalized_samples = self._unnormalize_samples_from_design_space(
            normalized_samples, design_space)
        samples = self._reformat_samples_from_design_space(
            unnormalized_samples, design_space)
        samples_df = self._put_samples_in_df_format(samples, design_space)

        return samples_df

    def _generate_normalized_samples(self, sampling_algo_name, algo_options, design_space):
        '''
        Method that generate normalized samples

        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm
            design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs

        Returns:
            normalized_samples (numpy matrix) :   matrix of n raws  (each raw is an input point to be evaluated)  
                                                  any variable of dim m will be in m columns of the matrix 
        '''
        gemseo_options = self._generate_gemseo_options(
            algo_options, design_space)

        normalized_samples = self._generate_normalized_samples_from_doe_factory(
            sampling_algo_name, **gemseo_options)  # call to gemseo
        return normalized_samples

    def _generate_gemseo_options(self, algo_options, design_space):
        """
        Providing algorithm's option in format needed for the _generate_samples method of the Doe Factory. 
        Those options comes:
        - from algo_options dict
        - from design space: dimension, variables_names, variables_sizes 
                  - dimension is not in algo options in gemseo because it is in the design space and it is needed in the doe selected algorithm
                  - variables_names, variables_sizes: only needed for DiagonalDOE algo and provided by design space


        Arguments:
            algo_options (dict): provides the selected value of each option of the algorithm
                                 each option can be either 'default' or with a user's selected value
            design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs    

        Returns:
             gemseo_options (dict): the gemseo options dict for _generate_samples method of the Doe Factory.
                                    It has options of the algorithm and the dimension of the design space. 
                                    It has also variables_names, and variables_sizes (for DiagonalDOE algo)

        """

        gemseo_options = {}
        for algo_option in algo_options:
            if algo_options[algo_option] != 'default':  # to be depreciated
                gemseo_options[algo_option] = algo_options[algo_option]

        self.logger.info(gemseo_options)
        # TODO : logging from module ?

        # The following 3 lines come from compute_doe in doe_lib.py of gemseo
        gemseo_options[self.DIMENSION] = design_space.dimension
        # Remark: those two following lines _VARIABLES_NAMES and _VARIABLES_SIZES are only used in gemseo
        # lib_scalable.py for DiagonalDOE algorithm and associated reverse
        # algo option.
        gemseo_options[self._VARIABLES_NAMES] = design_space.variables_names
        gemseo_options[self._VARIABLES_SIZES] = design_space.variables_sizes

        return gemseo_options

    def _generate_normalized_samples_from_doe_factory(self, sampling_algo_name, **gemseo_options):
        """
        Generating samples for the Doe using the _generate_samples method of the Doe Factory

        Arguments:
             gemseo_options (dict): the gemseo options dict for _generate_samples method of the Doe Factory.
                                    It has options of the algorithm and dimension of the design space. 
                                    It has also variables_names, and variables_sizes (for DiagonalDOE algo)

        Returns:
            normalized_samples (numpy matrix of floats) :  normalized_samples
                                                 matrix of n raws  (each raw is an input point to be evaluated)  
                                                 any variable of dim m will be in m columns of the matrix 

        """
#         doe_factory = DOEFactory()
        algo = self.doe_factory.create(sampling_algo_name)
        normalized_samples = algo._generate_samples(**gemseo_options)
        return normalized_samples

    def _unnormalize_samples_from_design_space(self, normalized_samples, design_space):
        """
        Un-normalize sample from design space lower and upper bound
        Check whether the variables satisfy the design space requirements
        It uses methods from gemseo Design Space

        Arguments:
            normalized_samples (numpy matrix of floats) :  matrix of n raws  (each raw is an input point to be evaluated)  
                                                  any variable of dim m will be in m columns of the matrix  
            design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs

        Returns:
            samples (numpy matrix of floats) : unnormalized samples    
                                     matrix of n raws  (each raw is an input point to be evaluated)  
                                     any variable of dim m will be in m columns of the matrix 


        Raises:
            ValueError: Either if the dimension of the values vector is wrong,
                if the values are not specified as an array or a dictionary,
                if the values are outside the bounds of the variables or
                if the component of an integer variable is an integer.
        """
        # the provided samples are normalized as bounds of design space where not
        # used yet
        unnormalize_vect = design_space.unnormalize_vect
        round_vect = design_space.round_vect
        samples = []
        for sample in normalized_samples:  # To be vectorized
            x_sample = round_vect(unnormalize_vect(sample))
            design_space.check_membership(x_sample)
            samples.append(x_sample)
        return samples

    def _reformat_samples_from_design_space(self, samples, design_space):
        """
        Reformat samples based on the design space to take into account variables with dim >1
        It uses methods from gemseo Design Space
        For instance in case of variables of x of dim 1 and z of dim 2 
        [0.0,-10.0, 0.0] becomes [[0.0]   [-10.0, 0.0]]

        Arguments:
            samples (numpy matrix of floats) : matrix of n raws  (each raw is an input point to be evaluated)  
                                     any variable of dim m will be an array of dim m in a single column of the matrix 
            selected_inputs (list): list of selected variables (the true variables in eval_inputs Desc_in)
            design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs

        Returns:
            reformated_samples (numpy matrix of arrays) : Reformated samples that takes into account variables with dim >1
                                    matrix of n raws  (each raw is an input point to be evaluated)  
                                    any variable of dim m is an array of dim m in a single column of the matrix 
        """
        selected_inputs = list(design_space.keys())

        reformated_samples = []
        for current_point in samples:  # To be vectorized
            # Current point  is an array with variables ordered as in selected_inputs
            # Find the dictionary version of the current point sample
            current_point_dict = design_space.array_to_dict(current_point)

        
            reformated_current_point = []
            for in_variable in selected_inputs:

                # convert array into data when needed
                if in_variable in self.selected_inputs_types.keys() and (
                    self.selected_inputs_types[in_variable] in ['float', 'int', 'string']):
                    current_point_dict[in_variable] = current_point_dict[in_variable][0]
                if in_variable in self.selected_inputs_types.keys() and (
                    self.selected_inputs_types[in_variable] in ['list']):
                    current_point_dict[in_variable] = list(current_point_dict[in_variable])

                reformated_current_point.append(
                    current_point_dict[in_variable])
            reformated_samples.append(reformated_current_point)

        return reformated_samples
    
    def _put_samples_in_df_format(self, samples, design_space):
        """
        construction of a dataframe of the generated samples
        # To be vectorized

        Arguments:
            samples (numpy matrix of arrays) : matrix of n raws  (each raw is an input point to be evaluated)  
                                               any variable of dim m will be an array of dim m in a single column of the matrix
            design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs
        Returns:
            samples_df (data_frame) : dataframe of a matrix of n raws  (each raw is an input point to be evaluated)  
                                      any variable of dim m is an array of dim m in a single column of the matrix
        """
        selected_inputs = list(design_space.keys())

        samples_df = pd.DataFrame(data=samples,
                                  columns=selected_inputs)
        

        return samples_df


    # TODO: REFACTOR IF POSSIBLE W/O PROXY REFs (note for the moment proxy is the wrapper until config. actions moved)
    def setup(self, proxy):
        """
        Method that setup the doe_algo method
        """
        dynamic_inputs = {}
        dynamic_outputs = {}

        # Setup dynamic inputs in case of DOE_ALGO selection:
        # i.e. EVAL_INPUTS and SAMPLING_ALGO
        self.setup_dynamic_inputs_for_doe_generator_method(dynamic_inputs, proxy)
        # Setup dynamic inputs when EVAL_INPUTS/SAMPLING_ALGO are already set
        self.setup_dynamic_inputs_algo_options_design_space(dynamic_inputs, proxy)

        return dynamic_inputs, dynamic_outputs

    def setup_dynamic_inputs_for_doe_generator_method(self, dynamic_inputs, proxy):
        """
        Method that setup dynamic inputs in case of DOE_ALGO selection: i.e. EVAL_INPUTS and SAMPLING_ALGO
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated

        """
        if proxy.sampling_method == proxy.DOE_ALGO:
            # Get possible values for sampling algorithm name
            available_doe_algorithms = self.get_available_algo_names()
            dynamic_inputs.update({'sampling_algo':
                                       {proxy.TYPE: 'string',
                                        proxy.STRUCTURING: True,
                                        proxy.POSSIBLE_VALUES: available_doe_algorithms}
                                   })

    def setup_dynamic_inputs_algo_options_design_space(self, dynamic_inputs, proxy):
        """
            Setup dynamic inputs when EVAL_INPUTS/SAMPLING_ALGO are already set
            Create or update DESIGN_SPACE
            Create or update ALGO_OPTIONS
        """
        self.setup_design_space(dynamic_inputs, proxy)
        self.setup_algo_options(dynamic_inputs, proxy)

    def setup_design_space(self, dynamic_inputs, proxy):
        """
        Method that setup 'design_space'
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        """
        # TODO: might want to refactor to simplify GridSearch
        selected_inputs_has_changed = False
        disc_in = proxy.get_data_in()
        # Dynamic input of default design space
        if proxy.EVAL_INPUTS in disc_in:

            # save possible types in sample generator
            if proxy.eval_in_possible_types is not None:
                self.selected_inputs_types = proxy.eval_in_possible_types.copy()

            eval_inputs = proxy.get_sosdisc_inputs(proxy.EVAL_INPUTS)
            if eval_inputs is not None:
                selected_inputs = eval_inputs[eval_inputs['selected_input'] == True]['full_name'].tolist()

                if set(selected_inputs) != set(self.selected_inputs):
                    self.selected_inputs = selected_inputs
                    

                default_design_space = pd.DataFrame()
                design_space_dataframe_descriptor = {
                    self.VARIABLES: ('string', None, False),
                    self.LOWER_BOUND: ('multiple', None, True),
                    self.UPPER_BOUND: ('multiple', None, True),
                    self.VALUES: ('multiple', None, True),
                    self.ENABLE_VARIABLE_BOOL: ('bool', None, True),
                    self.LIST_ACTIVATED_ELEM: ('list', None, True),
                }

                if proxy.sampling_method == proxy.DOE_ALGO:
                    default_design_space = pd.DataFrame({self.VARIABLES: self.selected_inputs,
                                                         self.LOWER_BOUND: [None] * len(self.selected_inputs),
                                                         self.UPPER_BOUND: [None] * len(self.selected_inputs),
                                                         self.LIST_ACTIVATED_ELEM: [[]] * len(self.selected_inputs),
                                                         self.ENABLE_VARIABLE_BOOL: [False] * len(self.selected_inputs),
                                                         self.VALUES: [None] * len(self.selected_inputs),
                                                         })
                    default_design_space[self.ENABLE_VARIABLE_BOOL] = default_design_space[self.ENABLE_VARIABLE_BOOL].astype(bool)
                elif proxy.sampling_method == proxy.GRID_SEARCH:
                    default_design_space = pd.DataFrame({self.VARIABLES: self.selected_inputs,
                                                         self.LOWER_BOUND: [0.0] * len(self.selected_inputs),
                                                         self.UPPER_BOUND: [100.0] * len(self.selected_inputs),
                                                         self.NB_POINTS: [2] * len(self.selected_inputs),
                                                         self.LIST_ACTIVATED_ELEM: [[]] * len(self.selected_inputs),
                                                         self.ENABLE_VARIABLE_BOOL: [False] * len(self.selected_inputs),
                                                         self.VALUES: [None] * len(self.selected_inputs),
                                                         })
                    default_design_space[self.NB_POINTS] = default_design_space[self.NB_POINTS].astype(int)
                    design_space_dataframe_descriptor.update({self.NB_POINTS: ('int', None, True)})
                dynamic_inputs.update({proxy.DESIGN_SPACE: {proxy.TYPE: 'dataframe',
                                                            proxy.DEFAULT: default_design_space,
                                                            proxy.STRUCTURING: False,
                                                            proxy.DATAFRAME_DESCRIPTOR: design_space_dataframe_descriptor}})

                # Next lines of code treat the case in which eval inputs change with a previously defined design space,
                # so that the bound are kept instead of set to default None.
                if 'design_space' in disc_in:
                    disc_in['design_space'][proxy.DEFAULT] = default_design_space
                    proxy.dm.set_data(proxy.get_var_full_name(proxy.DESIGN_SPACE, disc_in),
                                      proxy.DATAFRAME_DESCRIPTOR, design_space_dataframe_descriptor, check_value=False)

                    from_design_space = list(
                        disc_in['design_space'][proxy.VALUE][self.VARIABLES])
                    from_eval_inputs = self.selected_inputs

                    df_cols = [self.VARIABLES, self.LOWER_BOUND, self.UPPER_BOUND] + (
                        [self.NB_POINTS] if proxy.sampling_method == proxy.GRID_SEARCH else []) + (
                        [self.LIST_ACTIVATED_ELEM, self.ENABLE_VARIABLE_BOOL, self.VALUES])
                    final_dataframe = pd.DataFrame(columns=df_cols)

                    for element in from_eval_inputs:
                        default_row = default_design_space[default_design_space[self.VARIABLES] == element].iloc[0]
                        final_dataframe = final_dataframe.append(default_row, ignore_index=True)
                        if element in from_design_space:
                            to_append = disc_in['design_space'][proxy.VALUE][disc_in['design_space'][proxy.VALUE][
                                                                          self.VARIABLES] == element]
                            # TODO: in the current implementation it would be more proper that GridSearch setup its
                            #  own design space instead of having particular cases in the Doe sample generator.
                            if proxy.sampling_method == proxy.DOE_ALGO:
                                # for DoE need to dismiss self.NB_POINTS
                                to_append = to_append.loc[:, to_append.columns != self.NB_POINTS]
                            elif proxy.sampling_method == proxy.GRID_SEARCH and self.NB_POINTS not in to_append.columns:
                                # for GridSearch need to eventually insert the self.NB_POINTS column
                                to_append.insert(3, self.NB_POINTS, 2)
                            final_dataframe.loc[len(final_dataframe)-1, to_append.columns] = to_append.iloc[0, :]
                    proxy.dm.set_data(proxy.get_var_full_name(proxy.DESIGN_SPACE, disc_in),
                                      proxy.VALUE, final_dataframe, check_value=False)

    def setup_algo_options(self, dynamic_inputs, proxy):
        """
            Method that setup 'algo_options'
            Arguments:
                dynamic_inputs (dict): the dynamic input dict to be updated
        """
        disc_in = proxy.get_data_in()
        # Dynamic input of algo_options
        if proxy.ALGO in disc_in:
            algo_name = proxy.get_sosdisc_inputs(proxy.ALGO)
            if algo_name is not None:  # and algo_name_has_changed:
                default_dict = self.get_algo_default_options(algo_name)
                algo_options_dict = {proxy.ALGO_OPTIONS: {proxy.TYPE: 'dict', proxy.DEFAULT: default_dict,
                                                         proxy.DATAFRAME_EDITION_LOCKED: False,
                                                         proxy.STRUCTURING: True,
                                                         proxy.DATAFRAME_DESCRIPTOR: {
                                                             self.VARIABLES: ('string', None, False),
                                                             self.VALUES: ('string', None, True)}}}
                dynamic_inputs.update(algo_options_dict)
                all_options = list(default_dict.keys())
                if proxy.ALGO_OPTIONS in disc_in and disc_in[proxy.ALGO_OPTIONS][proxy.VALUE] is not None and list(
                        disc_in[proxy.ALGO_OPTIONS][proxy.VALUE].keys()) != all_options:
                    options_map = ChainMap(
                        disc_in[proxy.ALGO_OPTIONS][proxy.VALUE], default_dict)
                    disc_in[proxy.ALGO_OPTIONS][proxy.VALUE] = {
                        key: options_map[key] for key in all_options}

    def get_algo_default_options(self, algo_name):
        """
            This algo generate the default options to set for a given doe algorithm
        """
        # In get_options_and_default_values, it is already checked whether the algo_name belongs to the list of possible Gemseo
        # DoE algorithms
        if algo_name in get_available_doe_algorithms():
            algo_options_desc_in, algo_options_descr_dict = self.get_options_and_default_values(
                algo_name)
            return algo_options_desc_in
        else:
            raise Exception(
                f"A DoE algorithm which is not available in GEMSEO has been selected.")

    def get_arguments(self, wrapper):
        # Dynamic input of default design space
        algo_name = wrapper.get_sosdisc_inputs(wrapper.ALGO)
        algo_options = wrapper.get_sosdisc_inputs(wrapper.ALGO_OPTIONS)
        dspace_df = wrapper.get_sosdisc_inputs(wrapper.DESIGN_SPACE)
        eval_inputs = wrapper.get_sosdisc_inputs(wrapper.EVAL_INPUTS)
        selected_inputs = eval_inputs[eval_inputs[wrapper.SELECTED_INPUT] == True][wrapper.FULL_NAME].tolist()
        design_space = self.create_design_space(selected_inputs, dspace_df)
        doe_kwargs = {'sampling_algo_name': algo_name,
                      'algo_options': algo_options,
                      'design_space': design_space}
        return [], doe_kwargs

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
            design_space, _ = dspace_tool.create_gemseo_dspace_from_dspace_df(dspace_df_updated)
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

        # FIXME: why are we dismissing some of the user-input values in design_space ?
        enable_variables = [True for _ in selected_inputs]
        dspace_df_updated = pd.DataFrame({self.VARIABLES: selected_inputs,
                                          self.VALUES: values,
                                          self.LOWER_BOUND: lower_bounds,
                                          self.UPPER_BOUND: upper_bounds,
                                          self.ENABLE_VARIABLE_BOOL: enable_variables,
                                          self.LIST_ACTIVATED_ELEM: [[True] for _ in selected_inputs]})
        return dspace_df_updated

    def is_ready_to_sample(self, proxy):
        disc_in = proxy.get_data_in()
        return self.selected_inputs and \
            proxy.ALGO in disc_in and proxy.ALGO_OPTIONS in disc_in and proxy.DESIGN_SPACE in disc_in

    def filter_inputs(self, proxy):
        """
        Filter for the majority of algorithms the
        """
        disc_in = proxy.get_data_in()
        if proxy.ALGO in disc_in and proxy.get_sosdisc_inputs(proxy.ALGO) in self.TYPE_PERMISSIVE_ALGORITHMS:
            pass
        elif proxy.eval_in_possible_types:
            proxy.eval_in_possible_types = {_v: _t for (_v, _t) in proxy.eval_in_possible_types.items() if _t in ('array', 'float')}
            proxy.eval_in_possible_values = [_v for _v in proxy.eval_in_possible_values if _v in proxy.eval_in_possible_types]