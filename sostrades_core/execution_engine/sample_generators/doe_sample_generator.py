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
from builtins import NotImplementedError

from sostrades_core.execution_engine.sample_generators.abstract_sample_generator import AbstractSampleGenerator

from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.api import get_available_doe_algorithms
from gemseo.api import get_algorithm_options_schema
from gemseo.api import compute_doe

from tqdm import tqdm
import pandas as pd
from gemseo.utils.source_parsing import get_options_doc

import logging
LOGGER = logging.getLogger(__name__)

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''


class SampleTypeError(TypeError):
    pass


class DoeSampleGenerator(AbstractSampleGenerator):
    '''
    Abstract class that generates sampling
    '''

    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    N_PROCESSES = 'n_processes'
    WAIT_TIME_BETWEEN_SAMPLES = 'wait_time_between_samples'

    N_SAMPLES = "n_samples"

    # def __init__(self, generator_name):
    #     '''
    #     Constructor
    #     '''
    #     self.name = generator_name

    def get_available_algo_names(self):
        '''
        Method that provides the list of available algo_names

        Returns:
            the Sample Generator expected inputs (as DESC_IN format)
                                (to be provided to proxy i/o grammars)
            More precisely:
             algo_names_list (list): list of available algo names
        '''
        return get_available_doe_algorithms()

    def _check_algo_name(self, sampling_algo_name):
        '''
        Check provided algo name before getting its algo_options
        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
        Raises:
            Exception if sampling_algo_name is not in the list of available algorithms
        '''
        algo_names_list = self.get_available_algo_names()
        if sampling_algo_name not in algo_names_list:
            raise Exception(
                f"The provided algorithm name {sampling_algo_name} is not in the available algorithm list : {algo_names_list}")
        elif sampling_algo_name == 'CustomDOE':
            raise Exception(
                f"The provided algorithm name {sampling_algo_name} is not allowed in doe sample generator")

    def get_options_desc_in(self, sampling_algo_name):
        '''
        Method that provides the list of options of an algorithm with there default values (if any) and description

        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
        Returns:
            the Sample Generator expected inputs (as DESC_IN format)
                                (to be provided to proxy i/o grammars)
            More precisely:
             algo_options_desc_in (dict): dict of algo options with default values (if any). It is in algo_options desci_in format
             algo_options_descr_dict (dict): dict of description of algo options
             e.g. https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html#fullfact

        '''

        # check sampling_algo_name
        self._check_algo_name(sampling_algo_name)

        # get options of the sampling_algo_name in desc_in format
        doe_factory = DOEFactory()
        algo_lib = doe_factory.create(sampling_algo_name)

        # Remark: The following lines of code should be in gemseo
        # We should use only one line or two provided by gemseo
        fn = algo_lib.__class__._get_options

        algo_options_descr_dict = get_options_doc(fn)

        def get_default_args(func):
            import inspect
            signature = inspect.signature(func)
            return {
                k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty
            }

        algo_options_desc_in = get_default_args(fn)

        return algo_options_desc_in, algo_options_descr_dict

    def _check_options(self, sampling_algo_name, algo_options):
        '''
        Check provided options before sample generation
        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm        
        '''

        if self.N_SAMPLES not in algo_options:
            LOGGER.warning("N_samples is not defined; pay attention you use fullfact algo "
                           "and that levels are well defined")

        pass

    def generate_samples(self, sampling_algo_name, algo_options, selected_inputs, design_space):
        '''
        Method that generate samples in a design space for a selected algorithm with its options 
        The method also checks the output formating


        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm 
            selected_inputs (list): list of selected variables (the true variables in eval_inputs)
            design_space (gemseo DesignSpace): Design Space

        Returns:
            samples_df (dataframe) : generated samples              
        '''
        # check options
        self._check_options(sampling_algo_name, algo_options)

        # generate the sampling by subclass
        samples_df = self._generate_samples(
            sampling_algo_name, algo_options, selected_inputs, design_space)

        # check sample formatting
        # self._check_samples(samples)

        return samples_df

    def _check_samples(self, samples):
        '''
        Method that checks the sample output type
        Arguments:
            samples () : 
        Raises:
            Exception if xxxx                   
        '''
        if not(type(samples) is list):
            msg = "Expected sampling output type should be <list>, "
            msg += "however sampling type of sampling generator <%s> " % str(
                self.__class__.__name__)
            msg += "is <%s> " % str(type(samples))
            raise SampleTypeError()

    def _generate_samples(self, sampling_algo_name, algo_options, selected_inputs, design_space):
        '''
        Method that generate samples

        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm
            selected_inputs (list): list of selected variables (the true variables in eval_inputs)
            design_space (gemseo DesignSpace): Design Space

        Returns:
            samples_df (dataframe) : generated samples
        '''
        normalized_samples = self._generate_normalized_samples(
            sampling_algo_name, algo_options, design_space)
        unnormalized_samples = self.unnormalized_samples_from_design_space(
            normalized_samples, design_space)
        samples = self.reformat_samples_from_design_space(
            unnormalized_samples, selected_inputs, design_space)
        samples_df = self.put_samples_in_df_format(samples, selected_inputs)

        # return samples
        return samples_df

    def _generate_normalized_samples(self, sampling_algo_name, algo_options, design_space):
        '''
        Method that generate normalized samples

        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm
            design_space (gemseo DesignSpace): Design Space

        Returns:
            normalized_samples () :  
        '''
        gemseo_options = self.generate_gemseo_options(
            algo_options, design_space)

        normalized_samples = self.generate_normalized_samples_from_doe_factory(
            sampling_algo_name, **gemseo_options)  # call to gemseo
        return normalized_samples

    def generate_gemseo_options(self, algo_options, design_space):
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
            design_space (gemseo DesignSpace): Design Space      

        Returns:
             gemseo_options (dict): the gemseo options dict for _generate_samples method of the Doe Factory.
                                    It has options of the algorithm and the dimension of the design space. 
                                    It has also variables_names, and variables_sizes (for DiagonalDOE algo)

        """

        gemseo_options = {}
        for algo_option in algo_options:
            if algo_options[algo_option] != 'default':  # to be depreciated
                gemseo_options[algo_option] = algo_options[algo_option]

        LOGGER.info(gemseo_options)
        # TODO : logging from module ?

        gemseo_options[self.DIMENSION] = design_space.dimension
        gemseo_options[self._VARIABLES_NAMES] = design_space.variables_names
        gemseo_options[self._VARIABLES_SIZES] = design_space.variables_sizes
        # This comes from compute_doe in doe_lib.py of gemseo

        # Remark: _VARIABLES_NAMES and _VARIABLES_SIZES are only used in gemseo
        # lib_scalable.py for DiagonalDOE algorithm and associated reverse
        # algo option.

        return gemseo_options

    def generate_normalized_samples_from_doe_factory(self, sampling_algo_name, **gemseo_options):
        """
        Generating samples for the Doe using the _generate_samples method of the Doe Factory

        Arguments:
             gemseo_options (dict): the gemseo options dict for _generate_samples method of the Doe Factory.
                                    It has options of the algorithm and dimension of the design space. 
                                    It has also variables_names, and variables_sizes (for DiagonalDOE algo)

        Returns:
            normalized_samples (numpy.ndarray) :  matrix of n raws  (each raw is an input point to be evaluated)  
                                                  any variable of dim m will be in m columns of the matrix 

        """
        doe_factory = DOEFactory()
        algo = doe_factory.create(sampling_algo_name)
        normalized_samples = algo._generate_samples(**gemseo_options)
        return normalized_samples

    def unnormalized_samples_from_design_space(self, normalized_samples, design_space):
        """
        Un-normalized sample from design space lower and upper bound
        Check whether the variables satisfy the design space requirements
        It uses methods from gemseo Design Space

        Arguments:
            normalized_samples () : 
            design_space (gemseo DesignSpace): Design Space

        Returns:
            samples () : unnormalized samples

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

    def reformat_samples_from_design_space(self, samples, selected_inputs, design_space):
        """
        Reformat samples based on the design space to take into account variables with dim >1
        It uses methods from gemseo Design Space

        Arguments:
            samples () : unnormalized samples
            selected_inputs (list): list of selected variables (the true variables in eval_inputs)
            design_space (gemseo DesignSpace): Design Space

        Returns:
            reformated_samples () : Reformated samples that takes into account variables with dim >1
        """
        reformated_samples = []
        for current_point in samples:  # To be vectorized
            # Current point  is an array with variables ordered as in selected_inputs
            # Find the dictionary version of the current point sample
            current_point_dict = design_space.array_to_dict(current_point)

            # FIXME : are conversions needed here?
            # sample_dict = self._convert_array_into_new_type(sample_dict)

            # We reconstruct the current point as an array with variables
            # ordered as in selected_inputs
            reformated_current_point = []
            for in_variable in selected_inputs:
                reformated_current_point.append(
                    current_point_dict[in_variable])
            reformated_samples.append(reformated_current_point)

        return reformated_samples

    def put_samples_in_df_format(self, samples, selected_inputs):
        """
        construction of a dataframe of the generated samples
        # To be vectorized

        Arguments:
            samples () : 
            selected_inputs (list): list of selected variables (the true variables in eval_inputs)

        Returns:
            samples_df (data_frame) :
        """
        samples_df = pd.DataFrame(data=samples,
                                  columns=selected_inputs)
        return samples_df
