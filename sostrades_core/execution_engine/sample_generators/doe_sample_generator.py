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

    def __init__(self, generator_name):
        '''
        Constructor
        '''
        self.name = generator_name

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

    def _check_algo_name(self, algo_name):
        '''
        Check provided algo name before getting its algo_options
        Arguments:
            algo_name (string): name of the numerical algorithm
        Raises:
            Exception if algo_name is not in the list of available algorithms
        '''
        algo_names_list = self.get_available_algo_names()
        if algo_name not in algo_names_list:
            raise Exception(
                f"The provided algorithm name {algo_name} is not in the available algorithm list : {algo_names_list}")

    def get_options_desc_in(self, algo_name):
        '''
        Method that provides the list of options of an algorithm with there default values (if any) and description

        Arguments:
            algo_name (string): name of the numerical algorithm
        Returns:
            the Sample Generator expected inputs (as DESC_IN format)
                                (to be provided to proxy i/o grammars)
            More precisely:
             algo_options_desc_in (dict): dict of algo options with default values (if any). It is in algo_options desci_in format
             algo_options_descr_dict (dict): dict of description of algo options
             e.g. https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html#fullfact

        '''

        # check algo_name
        self._check_algo_name(algo_name)

        # get options of the algo_name in desc_in format
        doe_factory = DOEFactory()
        algo_lib = doe_factory.create(algo_name)

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

    def _check_options(self, algo_name, algo_options):
        '''
        Check provided options before sample generation
        Arguments:
            algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm        
        '''

        if self.N_SAMPLES not in algo_options:
            LOGGER.warning("N_samples is not defined; pay attention you use fullfact algo "
                           "and that levels are well defined")

        pass

    def generate_samples(self, algo_name, algo_options, eval_in_list, design_space):
        '''
        Method that generate samples in a design space for a selected algorithm with its options 
        The method also checks the output formating


        Arguments:
            algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm 
            eval_in_list (list): list of selected variables
            design_space (DesignSpace): Design Space

        Returns:
            samples () :               
        '''
        # check options
        self._check_options(algo_name, algo_options)

        # generate the sampling by subclass
        samples = self._generate_samples(
            algo_name, algo_options, eval_in_list, design_space)

        # check sample formatting
        # self._check_samples(samples)

        return samples

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

    def _generate_samples(self, algo_name, algo_options, eval_in_list, design_space):
        '''
        Method that generate samples

        Arguments:
            algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm
            eval_in_list (list): list of selected variables
            design_space (DesignSpace): Design Space

        Returns:
            samples () : 
        '''
        normalized_samples = self._generate_normalized_samples(
            algo_name, algo_options, design_space)
        samples = self.prepare_samples_for_evaluation(
            normalized_samples, eval_in_list, design_space)

        return samples

    def _generate_normalized_samples(self, algo_name, algo_options, design_space):
        '''
        Method that generate normalized samples

        Arguments:
            algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm
            design_space (DesignSpace): Design Space

        Returns:
            normalized_samples () :  
        '''
        gemseo_options = self.generate_gemseo_options(
            algo_options, design_space)

        normalized_samples = self.generate_normalized_samples_from_doe_factory(
            algo_name, **gemseo_options)  # call to gemseo
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
            design_space (DesignSpace): Design Space      

        Returns:
             gemseo_options (dict): the gemseo options dict for _generate_samples method of the Doe Factory.
                                    It has options of the algorithm and the dimension of the design space. 
                                    It has also variables_names, and variables_sizes (for DiagonalDOE algo)

        """

        gemseo_options = {}
        for algo_option in algo_options:
            if algo_options[algo_option] != 'default':
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

    def generate_normalized_samples_from_doe_factory(self, algo_name, **gemseo_options):
        """
        Generating samples for the Doe using the _generate_samples method of the Doe Factory

        Arguments:
             gemseo_options (dict): the gemseo options dict for _generate_samples method of the Doe Factory.
                                    It has options of the algorithm and dimension of the design space. 
                                    It has also variables_names, and variables_sizes (for DiagonalDOE algo)

        Returns:
            samples (numpy.ndarray) :  matrix of n raws  (each raw is an input point to be evaluated)   

        """
        doe_factory = DOEFactory()
        algo = doe_factory.create(algo_name)
        normalized_samples = algo._generate_samples(**gemseo_options)
        return normalized_samples


####################################
    def prepare_samples_for_evaluation(self, normalized_samples, eval_in_list, design_space):
        """
        xxxx

        Arguments:
            samples () : 
            eval_in_list (list): list of selected variables
            design_space (DesignSpace): Design Space

        Returns:
            prepared_samples () :

        """
        updated_samples = self.update_samples_from_design_space(
            normalized_samples, design_space)
        prepared_samples = self.reformat_samples(
            updated_samples, eval_in_list, design_space)
        return prepared_samples

    def update_samples_from_design_space(self, normalized_samples, design_space):
        """
        xxxx

        Arguments:
            samples () : 
            design_space (DesignSpace): Design Space

        Returns:
            updated_sampless () :

        """
        # the provided samples are normalised as bounds of design space where not
        # used yet
        unnormalize_vect = design_space.unnormalize_vect
        round_vect = design_space.round_vect
        updated_samples = []
        for sample in normalized_samples:
            x_sample = round_vect(unnormalize_vect(sample))
            design_space.check_membership(x_sample)
            updated_samples.append(x_sample)
        return updated_samples

    def reformat_samples(self, samples, eval_in_list, design_space):
        """
        xxxx

        Arguments:
            samples () : 
            eval_in_list (list): list of selected variables
            design_space (DesignSpace): Design Space

        Returns:
            reformated_samples () :
        """
        reformated_samples = []
        for sample in samples:
            sample_dict = design_space.array_to_dict(sample)
            # FIXME : are conversions needed here?
            # sample_dict = self._convert_array_into_new_type(sample_dict)
            ordered_sample = []
            for in_variable in eval_in_list:
                ordered_sample.append(sample_dict[in_variable])
            reformated_samples.append(ordered_sample)
        return reformated_samples
