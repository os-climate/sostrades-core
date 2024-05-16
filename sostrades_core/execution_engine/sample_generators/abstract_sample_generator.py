'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/17-2024/05/16 Copyright 2023 Capgemini

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
from builtins import NotImplementedError

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''


class SampleTypeError(TypeError):
    pass


class AbstractSampleGenerator(object):
    '''
    Abstract class that generates sampling
    '''

    def __init__(self, generator_name, logger=logging.Logger):
        '''
        Constructor
        '''
        self.generator_name = generator_name
        self.logger = logger

    def generate_samples(self, *args, **kwargs):
        '''
        Method that generate samples and checks the output formating
        '''
        # check options
        self._check_options(*args, **kwargs)
        # generate the sampling by subclass
        samples = self._generate_samples(*args, **kwargs)
        # check sample formatting
        self._check_samples(samples)

        return samples

    def _generate_samples(self, *args, **kwargs):
        '''
        Method that generate samples
        To be overloaded by subclass
        '''
        raise NotImplementedError

    def _check_samples(self, samples):
        '''
        Method that checks the sample output type
        '''
        if not isinstance(samples, list):
            msg = "Expected sampling output type should be <list>, "
            msg += "however sampling type of sampling generator <%s> " % str(
                self.__class__.__name__)
            msg += "is <%s> " % str(type(samples))
            raise SampleTypeError()

    # def get_options_and_default_values(self, algo_name):
    #     '''
    #     Returns the Sample Generator expected inputs for the algo options of the selected algorithm
    #     (to be provided to proxy i/o grammars)
    #     To be overloaded by subclass
    #     '''
    #     raise NotImplementedError

    def _check_options(self, *args, **kwargs):
        '''
        Check provided options before sample generation
        To be overloaded by subclass
        '''
        pass

    def setup(self, proxy):
        """
        Method used in combination with the ProxySampleGenerator in order to configure a given generation method.
        Arguments:
            proxy (ProxySampleGenerator) : associated proxy discipline
        Returns:
            dynamic_inputs (dict) : dynamic inputs of the sample generation method
            dynamic_outputs (dict( : dynamic outputs of the sample generation method
        """
        return {}, {}

    # TODO: renaming proxy / wrapper / proxy_or_wrapper for clarity when impl. is fixed in next 2 methods
    def sample(self, wrapper):
        """
        Method that takes the wrapper as input and returns the output of generate_samples.
        """
        _args, _kwargs = self.get_arguments(wrapper)
        return self.generate_samples(*_args, **_kwargs)

    def get_arguments(self, wrapper):
        """
        Returns the Sample Generator expected inputs for the algo options of the selected algorithm
        (to be provided to proxy i/o grammars)
        To be overloaded by subclass
        """
        return [], {}

    def is_ready_to_sample(self, proxy):
        """
        Method that takes the ProxySampleGenerator as input and returns whether the configuration sequence is ready for
        sample generation, notably to avoid asking for sample generation before the corresponding inputs are added and
        loaded in the dm.
        """
        return True

    def filter_inputs(self, proxy):
        """
        Method that takes the ProxySampleGenerator as input and filters the possible evaluated inputs values and types
        in order to constrain the input for specific sample generators.
        """
        # proxy.eval_in_possible_values = [subprocess_inputs]
        # proxy.eval_in_possible_types = {subprocess_input: variable_type}
        pass