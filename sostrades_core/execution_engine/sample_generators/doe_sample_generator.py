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

    def generate_samples(self, algo_name, algo_options, n_processes, wait_time_between_fork,  eval_in_list, design_space):
        # def generate_samples(self, algo_name, algo_options, n_processes,
        # wait_time_between_fork,  eval_in_list, design_space):
        '''
        Method that generate samples and checks the output formating
        '''
        self.get_options(algo_name)
        # check options
        self._check_options(algo_name, algo_options)
        # generate the sampling by subclass
        samples = self._generate_samples(
            algo_name, algo_options, n_processes, wait_time_between_fork,  eval_in_list, design_space)
        # check sample formatting
        # self._check_samples(samples)

        return samples

    # def _generate_samples(self, **kwargs):
    def _generate_samples(self, algo_name, algo_options, n_processes, wait_time_between_fork,  eval_in_list, design_space):
        '''
        Method that generate samples
        To be overloaded by subclass
        '''
        filled_options = self.generate_filled_options_for_sample_generation(
            algo_options, n_processes, wait_time_between_fork, design_space)

        samples = self.generate_samples_from_doe_factory(
            algo_name, **filled_options)  # call to gemseo
        return samples

    def _check_samples(self, samples):
        '''
        Method that checks the sample output type
        '''
        if not(type(samples) is list):
            msg = "Expected sampling output type should be <list>, "
            msg += "however sampling type of sampling generator <%s> " % str(
                self.__class__.__name__)
            msg += "is <%s> " % str(type(samples))
            raise SampleTypeError()

    def get_options(self, algo_name):
        '''
        Returns the Sample Generator expected inputs (as DESC_IN format)
        (to be provided to proxy i/o grammars)
        To be overloaded by subclass
        '''
        doe_factory = DOEFactory()
        algo_lib = doe_factory.create(algo_name)
        opts_gram = algo_lib.init_options_grammar(algo_name)

        descr_dict = get_options_doc(algo_lib.__class__._get_options)
        option_keys = list(descr_dict.keys())
        option_keys_dict = {}
        for key in option_keys:
            option_keys_dict[key] = descr_dict[key]

        # algo._get_options()
        # algo.get_algorithm_options()
        #list_algo_names = get_available_doe_algorithms()

        #option_keys_dict = {}
        # option_keys = list(get_algorithm_options_schema(
        #    algo_name)['properties'].keys())
        # https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html#fullfact
        # for key in option_keys:
        #    option_keys_dict[key] = get_algorithm_options_schema(algo_name)[
        #       'properties'][key]

        return descr_dict

    # def _check_options(self, **kwargs)
    def _check_options(self, algo_name, algo_options):
        '''
        Check provided options before sample generation
        To be overloaded by subclass
        '''

        if self.N_SAMPLES not in algo_options:
            LOGGER.warning("N_samples is not defined; pay attention you use fullfact algo "
                           "and that levels are well defined")

        pass

    def generate_filled_options_for_sample_generation(self, algo_options, n_processes, wait_time_between_fork, design_space):
        """Generating samples for the Doe using the Doe Factory
        """

        filled_options = {}
        for algo_option in algo_options:
            if algo_options[algo_option] != 'default':
                filled_options[algo_option] = algo_options[algo_option]

        LOGGER.info(filled_options)
        # TODO : logging from module ?

        filled_options[self.DIMENSION] = design_space.dimension
        filled_options[self._VARIABLES_NAMES] = design_space.variables_names
        filled_options[self._VARIABLES_SIZES] = design_space.variables_sizes
        # filled_options[self.N_PROCESSES] = int(filled_options['n_processes'])
        filled_options[self.N_PROCESSES] = n_processes
        filled_options[self.WAIT_TIME_BETWEEN_SAMPLES] = wait_time_between_fork
        return filled_options

    def generate_samples_from_doe_factory(self, algo_name, **filled_options):
        """
        """
        doe_factory = DOEFactory()
        algo = doe_factory.create(algo_name)
        samples = algo._generate_samples(**filled_options)
        return samples

    def prepare_samples_for_evaluation(self, samples, eval_in_list, design_space):
        """
        """
        updated_samples = self.update_samples_from_design_space(
            samples, design_space)
        prepared_samples = self.reformat_samples(
            updated_samples, design_space, eval_in_list)
        return prepared_samples

    def update_samples_from_design_space(self, samples, design_space):
        """
        """
        unnormalize_vect = design_space.unnormalize_vect
        round_vect = design_space.round_vect
        updated_samples = []
        for sample in samples:
            x_sample = round_vect(unnormalize_vect(sample))
            design_space.check_membership(x_sample)
            updated_samples.append(x_sample)
        return updated_samples

    def reformat_samples(self, samples, design_space, eval_in_list):
        """

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
