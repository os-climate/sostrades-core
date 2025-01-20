'''
Copyright 2023 Capgemini

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

from gemseo import create_design_space, create_scenario
from gemseo.settings.doe import CustomDOE_Settings
from numpy import hstack, size
from pandas import DataFrame, concat

from sostrades_core.execution_engine.disciplines_wrappers.driver_evaluator_wrapper import (
    DriverEvaluatorWrapper,
)
from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import (
    SampleGeneratorWrapper,
)
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling


class MonoInstanceDriverWrapper(DriverEvaluatorWrapper):
    """Class that executes a DOE."""

    def prepare_input_samples(self) -> tuple[DataFrame, list[str]]:
        """Prepare the dataframe of input samples to evaluate.

        Returns:
            A tuple composed of:
              - a dataframe containing the samples input values,
              - the list of scenario names corresponding to each sample.
        """
        samples_df = self.get_sosdisc_inputs(SampleGeneratorWrapper.SAMPLES_DF)

        input_full_names = [
            f"{self.attributes['driver_name']}.{col}"
            for col in samples_df.columns
            if col not in [SampleGeneratorWrapper.SCENARIO_NAME, SampleGeneratorWrapper.SELECTED_SCENARIO]
        ]
        input_columns_short_name = [
            col
            for col in samples_df.columns
            if col not in [SampleGeneratorWrapper.SCENARIO_NAME, SampleGeneratorWrapper.SELECTED_SCENARIO]
        ]

        # Keep only the selected scenarios
        samples_df = samples_df.loc[samples_df[SampleGeneratorWrapper.SELECTED_SCENARIO]]
        samples_df = samples_df.drop(SampleGeneratorWrapper.SELECTED_SCENARIO, axis='columns')

        # Rename the columns with full names
        samples = samples_df.rename(
            mapper={
                input_name: f"{self.attributes['driver_name']}.{input_name}" for input_name in input_columns_short_name
            },
            axis=1,
        )
        if "reference_scenario" not in list(samples[SampleGeneratorWrapper.SCENARIO_NAME]):
            # Add the reference_scenario if it has not already been added by a SampleGenerator or user
            reference_values = self.get_sosdisc_inputs(input_full_names, full_name_keys=True)
            if len(input_full_names) == 1:
                reference_values = [reference_values]
            reference_scenario = {input_full_names[i]: [reference_values[i]] for i in range(len(input_full_names))}
            reference_scenario[SampleGeneratorWrapper.SCENARIO_NAME] = 'reference_scenario'
            samples = concat(
                (samples, DataFrame(reference_scenario, index=[samples.shape[0]])),
                axis=0,
            )

        # Store the dataframe of input values with variable full names
        samples_inputs_df = samples.rename(
            mapper={
                f"{self.attributes['driver_name']}.{input_name}": input_name for input_name in input_columns_short_name
            },
            axis=1,
        )
        self.store_sos_outputs_values({'samples_inputs_df': samples_inputs_df})

        scenario_names = samples.pop(SampleGeneratorWrapper.SCENARIO_NAME).to_list()
        return samples, scenario_names

    def evaluate_samples(self, input_samples: DataFrame) -> DataFrame:
        """Evaluate the samples.

        Args:
            input_samples: The dataframe containing the samples input values.

        Returns:
            The dataframe of output values.
        """
        self._init_input_data()
        default_inputs = self._get_input_data({})
        self.attributes["sub_disciplines"][0].default_input_data = default_inputs

        n_processes = self.get_sosdisc_inputs('n_processes') or 1
        wait_time_between_samples = self.get_sosdisc_inputs('wait_time_between_fork') or 0.0
        outputs = self.attributes["eval_out_list"]

        # Create a DOE scenario to evaluate the samples
        ds = create_design_space()
        for var in input_samples.columns:
            ds.add_variable(var, size=size(input_samples.loc[0, var]))
        doe_scenario = create_scenario(
            disciplines=self.attributes["sub_disciplines"][0],
            objective_name=outputs,
            design_space=ds,
            name="sampling_DOE",
            scenario_type="DOE",
            formulation_name="DisciplinaryOpt",
        )
        # Reformat the input samples to be compliant with GEMSEO
        input_array = hstack(input_samples.to_numpy().ravel()).reshape(-1, ds.dimension)
        settings = CustomDOE_Settings(
            samples=input_array,
            n_processes=n_processes,
            wait_time_between_samples=wait_time_between_samples,
        )
        doe_scenario.execute(settings)
        outputs_array = next(iter(doe_scenario.to_dataset().to_dict_of_arrays()["functions"].values()))

        return DataFrame(data=outputs_array, columns=outputs)

    def process_output(self, evaluation_outputs: DataFrame, scenario_names: list[str]) -> None:
        """Process and store the sampling outputs.

        Args:
            evaluation_outputs: The results of the samples evaluation.
            scenario_names: The scenario names corresponding to each sample.
        """
        columns = [SampleGeneratorWrapper.SCENARIO_NAME, *evaluation_outputs.columns]
        evaluation_outputs[SampleGeneratorWrapper.SCENARIO_NAME] = scenario_names
        samples_output_df = DataFrame(evaluation_outputs, columns=columns)

        # save data of last execution i.e. reference values # TODO: do this  better in refacto doe
        subprocess_ref_outputs = {
            key: self.attributes['sub_disciplines'][0].io.data[key]
            for key in self.attributes['sub_disciplines'][0].output_grammar.names
            if not key.endswith(ProxyCoupling.NORMALIZED_RESIDUAL_NORM)
        }
        self.store_sos_outputs_values(subprocess_ref_outputs, full_name_keys=True)

        self.store_sos_outputs_values({'samples_outputs_df': samples_output_df})
        for dynamic_output, out_name in zip(self.attributes['eval_out_list'], self.attributes['eval_out_names']):
            dict_output = {
                r[SampleGeneratorWrapper.SCENARIO_NAME]: r[dynamic_output]
                for _, r in samples_output_df.iterrows()
            }
            self.store_sos_outputs_values({out_name: dict_output})

    def run(self) -> None:
        """
        Execute the DOE.

        Overloads the SoSEval method.
        """
        samples_input_df, scenario_names = self.prepare_input_samples()
        evaluation_outputs = self.evaluate_samples(samples_input_df)
        self.process_output(evaluation_outputs, scenario_names)
