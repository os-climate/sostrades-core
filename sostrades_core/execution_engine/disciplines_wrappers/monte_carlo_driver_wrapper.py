'''
Copyright 2025 Capgemini

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

from copy import deepcopy
from dataclasses import dataclass
from enum import auto
from typing import TYPE_CHECKING, Any, ClassVar

from gemseo import create_scenario
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.settings.doe import PYDOE_LHS_Settings
from gemseo.settings.formulations import DisciplinaryOpt_Settings
from gemseo.uncertainty import create_statistics
from numpy import atleast_1d, cumsum, size, split, sqrt
from pandas import DataFrame, concat
from strenum import LowercaseStrEnum

from sostrades_core.execution_engine.disciplines_wrappers.driver_evaluator_wrapper import DriverEvaluatorWrapper
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator

DistributionsArgsType = dict[str, dict[str, Any]]

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset
    from gemseo.typing import NumberArray

    from sostrades_core.execution_engine.sos_discipline import SoSDiscipline


@dataclass
class SoSInputNames:
    """The names of the input parameters."""

    BATCH_SIZE = "batch_size"
    """The batch size for the sampling when using std or cv stopping criterion."""

    INPUT_DISTRIBUTIONS = "input_distributions"
    """The distributions of each input variables."""

    N_PROCESSES = "n_processes"
    """The number of processes for the sampling."""

    N_SAMPLES = "n_samples"
    """The number of samples to evaluate."""

    TARGET_CV = "target_cv"
    """The target coefficients of variation for each component of the estimator."""

    TARGET_STD = "target_std"
    """The target standard deviations for each component of the estimator."""

    WAIT_TIME_BETWEEN_SAMPLES = "wait_time_between_samples"
    """The time to wait between evaluationg two samples."""


@dataclass
class SoSOutputNames:
    """The names of the output parameters."""

    INPUT_SAMPLES = "input_samples"
    """The dictionary of input samples."""

    OUTPUT_SAMPLES = "output_samples"
    """The dictionary of output samples."""


class StoppingCriterion(LowercaseStrEnum):
    """The stopping criterion for the Monte Carlo sampling."""

    N_SAMPLES = auto()
    """Stop the sampling after a given number of samples."""

    STD = auto()
    """Stop the sampling when the standard deviation of the estimator reaches the target value,
    or when the maximum number of samples is reached."""

    CV = auto()
    """Stop the sampling when the coefficient of variation of the estimator reaches the target value,
    or when the maximum number of samples is reached."""


class MonteCarloDriverWrapper(DriverEvaluatorWrapper):
    """
    A driver that performs quasi-Monte Carlo sampling on quantities of interest.

    The sampling is a Randomized Quasi Monte Carlo of the uncertain space.
    """

    DISTRIBUTION_TYPE_KEY = "distribution_type"
    """The key of the distribution dictionary containing the distribution type."""

    SoSInputNames: ClassVar[type[SoSInputNames]] = SoSInputNames
    """The names of the driver inputs passed by SoSTrades."""

    SoSOutputNames: ClassVar[type[SoSOutputNames]] = SoSOutputNames
    """The names of the driver outputs passed to SoSTrades."""

    StoppingCriterion: ClassVar[type[StoppingCriterion]] = StoppingCriterion
    """The available stopping criteria."""

    parameter_space: ParameterSpace
    """The parameter space containing the distributions used for sampling."""

    outputs: list[str]
    """The list of output variable full names."""

    _discipline: SoSDiscipline
    """The discipline or MDA used to evaluate the samples."""

    _n_samples: int | None
    """The number of samples to compute when the stopping criterion is `n_samples`,
    or the maximum number of samples when the stopping criterion is `std` or `cv`."""

    _stopping_criterion: StoppingCriterion
    """The selected stopping criterion."""

    _target_cv: NumberArray | None
    """The target coefficients of variation for each component of the estimator."""

    _target_std: float | NumberArray | None
    """The target standard deviations for each component of the estimator."""

    def _setup_discipline(self):
        """Get the evaluation discipline and set the default input values."""
        self._discipline = self.attributes["sub_disciplines"][0]
        self._init_input_data()
        self._discipline.default_input_data = self._get_input_data({})

    def _setup_inputs_outputs(self) -> None:
        """Setup the input distributions and output names."""
        self.parameter_space = ParameterSpace()
        distributions_args: DistributionsArgsType = self.get_sosdisc_inputs(self.SoSInputNames.INPUT_DISTRIBUTIONS)
        for var_name, distr_args in distributions_args.items():
            # Copy the dict as it will be modified
            distr_arg_copy = deepcopy(distr_args)
            distributions_type = distr_arg_copy.pop(self.DISTRIBUTION_TYPE_KEY)
            self.parameter_space.add_random_variable(
                name=var_name,
                distribution=distributions_type,
                **distr_arg_copy,
            )

        gather_outputs = self.get_sosdisc_inputs(ProxyDriverEvaluator.GATHER_OUTPUTS)
        short_output_names = list(gather_outputs[gather_outputs["selected_output"]]["full_name"])
        self.outputs = [f"{self.attributes['driver_name']}.{name}" for name in short_output_names]

    def _setup_stopping_criterion(self) -> None:
        """Setup the stopping criterion."""
        self._n_samples = self.get_sosdisc_inputs(self.SoSInputNames.N_SAMPLES)
        if target_std := self.get_sosdisc_inputs(self.SoSInputNames.TARGET_STD):
            self._target_std = atleast_1d(target_std).flatten()
            self._stopping_criterion = StoppingCriterion.STD
        elif target_cv := self.get_sosdisc_inputs(self.SoSInputNames.TARGET_CV):
            self._target_cv = atleast_1d(target_cv).flatten()
            self._stopping_criterion = StoppingCriterion.CV
        else:
            self._stopping_criterion = StoppingCriterion.N_SAMPLES

    def _evaluate_n_samples(self, n_samples: int) -> Dataset:
        """
        Sample the input distributions for a given number of samples and evaluate the outputs.

        Args:
            n_samples: The number of samples to evaluate.

        Returns:
            The dataset of input and output values.

        """
        formulation_settings = DisciplinaryOpt_Settings()
        mc_scenario = create_scenario(
            disciplines=[self._discipline],
            objective_name=self.outputs,
            design_space=self.parameter_space,
            scenario_type="DOE",
            formulation_settings_model=formulation_settings,
        )
        n_processes = self.get_sosdisc_inputs(self.SoSInputNames.N_PROCESSES)
        wait_time_between_samples = self.get_sosdisc_inputs(self.SoSInputNames.WAIT_TIME_BETWEEN_SAMPLES)
        algo_settings = PYDOE_LHS_Settings(
            n_samples=n_samples,
            n_processes=n_processes,
            wait_time_between_samples=wait_time_between_samples,
        )
        mc_scenario.execute(algo_name="PYDOE_LHS", algo_settings_model=algo_settings)
        return mc_scenario.to_dataset()

    def _is_criterion_reached(self, dataset: Dataset) -> bool:
        """
        Check whether the stopping criterion is reached.

        Args:
            dataset: The dataset containing the input and output values.

        Returns:
            Whether the criterion is reached.

        """
        n = dataset.shape[0]
        analysis = create_statistics(dataset, variable_names=["_".join(self.outputs)])
        if self._stopping_criterion == StoppingCriterion.STD:
            std = next(iter(analysis.compute_standard_deviation().values()))
            return all(std / sqrt(n) <= self._target_std)
        cv = next(iter(analysis.compute_variation_coefficient().values()))
        return all(cv / sqrt(n) <= self._target_cv)

    def _compute_samples(self) -> Dataset:
        """
        Compute the Monte Carlo samples until the stopping criterion is reached.

        Returns:
            The dataset of input and output values.

        """
        if self._stopping_criterion == StoppingCriterion.N_SAMPLES:
            return self._evaluate_n_samples(self._n_samples)

        batch_size = self.get_sosdisc_inputs(self.SoSInputNames.BATCH_SIZE) or self._n_samples // 100
        dataset = self._evaluate_n_samples(batch_size)
        while not self._is_criterion_reached(dataset) and dataset.shape[0] < self._n_samples:
            dataset = concat((dataset, self._evaluate_n_samples(batch_size)))
        return dataset

    def _process_outputs(self, evaluation_outputs: Dataset) -> None:
        """
        Process and export the input and output values of the sampling.

        The output samples are gathered in a single array.
        We need to retrieve the size of each separate output
        and split the array into the sub-array corresponding to each output.

        Args:
            evaluation_outputs: The dataset containing the input and output values.

        """
        n_samples = evaluation_outputs.shape[0]
        samples_dict = evaluation_outputs.to_dict_of_arrays()
        input_dict = {k: v.flatten() for k, v in samples_dict["designs"].items()}
        self.store_sos_outputs_values({self.SoSOutputNames.INPUT_SAMPLES: DataFrame(input_dict)})
        output_array = next(iter(samples_dict["functions"].values()))
        output_sizes = [size(self.attributes["sub_disciplines"][0].local_data[output]) for output in self.outputs]
        if all(atleast_1d(output_sizes) == 1):  # all outputs have only 1 component
            samples_output_df = DataFrame(output_array, columns=self.outputs)
        else:  # some outputs have more than 1 component
            df_list = []
            output_arrays = split(output_array, cumsum(output_sizes[:-1]), axis=1)
            for i, a in enumerate(output_arrays):
                if output_sizes[i] == 1:
                    df = DataFrame({self.outputs[i]: a.flatten()})
                else:
                    df = DataFrame({self.outputs[i]: [a[j, :] for j in range(n_samples)]})
                df_list.append(df)
            samples_output_df = concat(df_list, axis=1)
        self.store_sos_outputs_values({self.SoSOutputNames.OUTPUT_SAMPLES: samples_output_df})

    def run(self) -> None:
        """
        Perform a Quasi Monte Carlo sampling of the selected output(s).

        Sample the input distribution and evaluate the outputs until the stopping criterion is reached.
        This criterion can be based on the number of samples,
        or on the standard deviation or coefficient of variation of the outputs.
        """
        # Setup the run
        self._setup_discipline()
        self._setup_inputs_outputs()
        self._setup_stopping_criterion()

        dataset = self._compute_samples()
        self._process_outputs(dataset)
