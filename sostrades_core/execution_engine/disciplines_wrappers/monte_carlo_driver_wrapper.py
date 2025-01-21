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

from typing import TYPE_CHECKING, Any, ClassVar

from gemseo import create_scenario
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.settings.doe import PYDOE_LHS_Settings
from gemseo.settings.formulations import DisciplinaryOpt_Settings
from gemseo.uncertainty import create_statistics
from numpy import atleast_1d, hstack
from pandas import DataFrame, concat
from strenum import StrEnum, auto

from sostrades_core.execution_engine.disciplines_wrappers.driver_evaluator_wrapper import DriverEvaluatorWrapper

DistributionsArgsType = dict[str, dict[str, Any]]

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset
    from gemseo.typing import NumberArray

    from sostrades_core.execution_engine.sos_discipline import SoSDiscipline


class StoppingCriterion(StrEnum):
    """The stopping criterion for the Monte Carlo sampling."""

    n_samples = auto()
    """Stop the sampling after a given number of samples."""

    std = auto()
    """Stop the sampling when the standard deviation of the estimator reaches the target value,
    or when the maximum number of samples is reached."""

    cv = auto()
    """Stop the sampling when the coefficient of variation of the estimator reaches the target value,
    or when the maximum number of samples is reached."""


class MonteCarloDriverWrapper(DriverEvaluatorWrapper):
    """A driver that performs quasi-Monte Carlo sampling on quantities of interest.

    The sampling is a Randomized Quasi Monte Carlo of the uncertain space.
    """

    StoppingCriterion: ClassVar[type[StoppingCriterion]] = StoppingCriterion
    """The available stopping criteria."""

    parameter_space: ParameterSpace
    """The parameter space containing the distributions used for sampling."""

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
    """The target standard variations for each component of the estimator."""

    def _setup_discipline(self):
        """Get the evaluation discipline and set the default input values."""
        self._discipline = self.attributes["sub_disciplines"][0]
        self._init_input_data()
        self._discipline.default_input_data = self._get_input_data({})

    def _setup_input_distributions(self) -> None:
        """Setup the GEMSEO input distributions for sampling."""
        self.parameter_space = ParameterSpace()

        distributions_args: DistributionsArgsType = self.get_sosdisc_inputs["input_distributions"]
        for var_name, distr_args in distributions_args.items():
            distributions_type = distr_args.pop("distribution_type")
            self.parameter_space.add_random_variable(
                name=var_name,
                distribution=distributions_type,
                **distributions_args,
            )

    def _setup_stopping_criterion(self) -> None:
        """Setup the stopping criterion."""
        self._n_samples = None
        self._target_cv = None
        self._target_std = None
        if n_samples := self.get_sosdisc_inputs("n_samples") is not None:
            self._n_samples = n_samples
            self._stopping_criterion = StoppingCriterion.n_samples
        elif target_std := self.get_sosdisc_inputs("target_std") is not None:
            self._target_std = atleast_1d(target_std).flatten()
            self._stopping_criterion = StoppingCriterion.std
        elif target_cv := self.get_sosdisc_inputs("target_cv") is not None:
            self._target_cv = atleast_1d(target_cv).flatten()
            self._stopping_criterion = StoppingCriterion.cv
        else:
            msg = f"Monte Carlo sampling requires a stopping criterion (`{StoppingCriterion.n_samples}`, `{StoppingCriterion.std}` or {StoppingCriterion.cv}`)."
            raise ValueError(msg)

    def _evaluate_n_samples(self, n_samples: int) -> Dataset:
        """Sample the input distributions for a given number of samples and evaluate the outputs.

        Args:
            n_samples: The number of samples to evaluate.

        Returns:
            The dataset of input and output values.
        """
        inputs = self.parameter_space.variable_names
        outputs = self.attributes["eval_out_list"]
        formulation_settings = DisciplinaryOpt_Settings()
        mc_scenario = create_scenario(
            disciplines=[self._discipline],
            objective_name=outputs,
            design_space=ParameterSpace,
            scenario_type="DOE",
            formulation_settings_model=formulation_settings,
        )
        n_processes = self.get_sosdisc_inputs("n_processes") or 1
        algo_settings = PYDOE_LHS_Settings(n_samples=n_samples, n_processes=n_processes)
        mc_scenario.execute(algo_name="PYDOE_LHS", algo_settings_model=algo_settings)
        # Keep only the request inputs and outputs
        return mc_scenario.to_dataset().loc[:, (slice(None), inputs + outputs)]

    def _is_criterion_reached(self, dataset: Dataset) -> bool:
        """Check whether the stopping criterion is reached.

        Args:
            dataset: The dataset containing the input and output values.

        Returns:
            Whether the criterion is reached.
        """
        analysis = create_statistics(dataset, variable_names=self.get_sosdisc_inputs["eval_out_list"])
        if self._stopping_criterion == StoppingCriterion.std:
            std = hstack(tuple(analysis.compute_standard_deviation()))
            return all(std <= self._target_std)
        cv = hstack(tuple(analysis.compute_variation_coefficient()))
        return all(cv <= self._target_std)

    def _compute_samples(self) -> Dataset:
        """Compute the Monte Carlo samples until the stopping criterion is reached.

        Returns:
            The dataset of input and output values.
        """
        if self._stopping_criterion == StoppingCriterion.n_samples:
            return self._evaluate_n_samples(self._n_samples)

        batch_size = self.get_sosdisc_inputs("batch_size") or self._n_samples // 10
        dataset = self._evaluate_n_samples(batch_size)
        while not self._is_criterion_reached(dataset) and dataset.shape[0] < self._n_samples:
            dataset = concat((dataset, self._evaluate_n_samples(batch_size)))
        return dataset

    def _process_outputs(self, dataset: Dataset) -> None:
        """Process and export the input and output values of the sampling.

        Args:
            dataset: The dataset containing the input and output values.
        """
        dict_of_arrays = dataset.to_dict_of_arrays()
        self.store_sos_outputs_values({"samples_inputs_df": DataFrame(dict_of_arrays["designs"])})
        self.store_sos_outputs_values({"samples_outputs_df": DataFrame(dict_of_arrays["functions"])})

    def run(self) -> None:
        """Perform a Quasi Monte Carlo sampling of the selected output(s).

        Sample the input distribution and evaluate the outputs until the stopping criterion is reached.
        This criterion can be based on the number of samples,
        or on the standard deviation or coefficient of variation of the outputs.
        """
        # Setup the run
        self._setup_discipline()
        self._setup_input_distributions()
        self._setup_stopping_criterion()

        dataset = self._compute_samples()
        self._process_outputs(dataset)
