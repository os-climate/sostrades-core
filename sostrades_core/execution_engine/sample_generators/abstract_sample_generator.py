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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

class SampleTypeError(TypeError):
    pass

class AbstractSampleGenerator(object):
    '''
    Abstract class that generates sampling
    '''

    def __init__(self, generator_name):
        '''
        Constructor
        '''
        self.name = generator_name
        
    def generate_samples(self, **kwargs):
        '''
        Method that generate samples and checks the output formating
        '''
        # generate the sampling by subclass
        samples = self._generate_samples(**kwargs)
        # check sample formatting
        self._check_samples(samples)
        
        return samples
        
    def _generate_samples(self, **kwargs):
        '''
        Method that generate samples
        To be overloaded by subclasses
        '''
        raise NotImplementedError
        
    def _check_samples(self, samples):
        '''
        Method that checks the sample output type
        '''
        if not(type(samples) is list):
            msg = "Expected sampling output type should be <list>, "
            msg += "however sampling type of sampling generator <%s> "%str(self.__class__.__name__)
            msg += "is <%s> "%str(type(samples))
            raise SampleTypeError()
