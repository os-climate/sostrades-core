'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/17-2025/02/14 Copyright 2025 Capgemini

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

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging

"""Abstract base class for sample generators."""


class SampleTypeError(TypeError):
    """Custom exception for sample type errors."""

    pass


class AbstractSampleGenerator(ABC):
    """Abstract class that generates sampling."""

    def __init__(self, generator_name: str, logger: logging.Logger) -> None:
        """
        Initialize sample generator.

        Args:
            generator_name: Name of the sample generator.
            logger: Logger instance for outputting messages.

        """
        self.generator_name = generator_name
        self.logger = logger

    def generate_samples(self, *args, **kwargs) -> list:
        """
        Generate samples and check output formatting.

        Args:
            *args: Variable arguments for sample generation.
            **kwargs: Keyword arguments for sample generation.

        Returns:
            List of generated samples.

        """
        # check options
        self._check_options(*args, **kwargs)
        # generate the sampling by subclass
        samples = self._generate_samples(*args, **kwargs)
        # check sample formatting
        self._check_samples(samples)

        return samples

    def _generate_samples(self, *args, **kwargs) -> list:
        """
        Generate samples (to be implemented by subclass).

        Args:
            *args: Variable arguments for sample generation.
            **kwargs: Keyword arguments for sample generation.

        Returns:
            List of generated samples.

        Raises:
            NotImplementedError: Must be implemented by subclass.

        """
        raise NotImplementedError

    def _check_samples(self, samples: list) -> None:
        """
        Check the sample output type.

        Args:
            samples: Generated samples to validate.

        Raises:
            SampleTypeError: If samples is not a list.

        """
        if not isinstance(samples, list):
            msg = "Expected sampling output type should be <list>, "
            msg += "however sampling type of sampling generator <%s> " % str(
                self.__class__.__name__)
            msg += "is <%s> " % str(type(samples))
            raise SampleTypeError(msg)

    # def get_options_and_default_values(self, algo_name):
    #     '''
    #     Returns the Sample Generator expected inputs for the algo options of the selected algorithm
    #     (to be provided to proxy i/o grammars)
    #     To be overloaded by subclass
    #     '''
    #     raise NotImplementedError

    def _check_options(self, *args, **kwargs) -> None:
        """
        Check provided options before sample generation.

        Args:
            *args: Variable arguments to validate.
            **kwargs: Keyword arguments to validate.

        Note:
            To be overloaded by subclass.

        """
        pass

    def setup(self, proxy) -> tuple[dict, dict]:
        """
        Configure sample generation method in combination with ProxySampleGenerator.

        Args:
            proxy: Associated proxy discipline.

        Returns:
            dynamic_inputs (dict) : dynamic inputs of the sample generation method
            dynamic_outputs (dict) : dynamic outputs of the sample generation method

        """
        return {}, {}

    def sample(self, wrapper) -> list:
        """
        Generate samples using wrapper input.

        Args:
            wrapper: Wrapper containing input data.

        Returns:
            Generated samples list.

        """
        _args, _kwargs = self.get_arguments(wrapper)
        return self.generate_samples(*_args, **_kwargs)

    def get_arguments(self, wrapper) -> tuple[list, dict]:
        """
        Get expected inputs for sample generation from wrapper.

        Args:
            wrapper: Wrapper containing generation parameters.

        Returns:
            Tuple of (args, kwargs) for sample generation.

        Note:
            To be overloaded by subclass.

        """
        return [], {}

    def is_ready_to_sample(self, proxy) -> bool:
        """
        Check if configuration is ready for sample generation.

        Args:
            proxy: ProxySampleGenerator instance.

        Returns:
            True if ready to generate samples, False otherwise.

        """
        return True

    def filter_inputs(self, proxy) -> None:
        """
        Filter possible evaluated inputs for specific sample generators.

        Args:
            proxy: ProxySampleGenerator instance to filter inputs for.

        """
        pass
