'''
Copyright 2024 Capgemini

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
import unittest
import pandas as pd

from sostrades_core.sos_processes.test.tests_driver_eval.mono.test_mono_driver_sample_generator_tornado_analysis.usecase_tornado_analysis import (
    Study,
)


class TestTornadoChartAnalysis(unittest.TestCase):
    """
    UncertaintyQuantification test class
    """

    def test_01_tornado_chart_analysis(self):

        ns = "usecase_tornado_analysis"

        input_selection_a = {
            "selected_input": [False, True, True],
            "full_name": ["x", "Coupling.Disc1.a", "Coupling.Disc1.b"],
        }
        input_selection_a = pd.DataFrame(input_selection_a)

        output_selection_ind = {
            "selected_output": [True, False, False],
            "full_name": ["y", "Coupling.Disc1.indicator", "y_array"],
        }
        output_selection_ind = pd.DataFrame(output_selection_ind)

        disc_dict = {}
        disc_dict[f"{ns}.SampleGenerator.sampling_method"] = "tornado_chart_analysis"
        disc_dict[f"{ns}.SampleGenerator.variation_list"] = [-10.0, 10.0]
        disc_dict[f"{ns}.Eval.with_sample_generator"] = True
        disc_dict[f"{ns}.SampleGenerator.eval_inputs"] = input_selection_a
        disc_dict[f"{ns}.Eval.gather_outputs"] = output_selection_ind

        # Disc1 inputs
        disc_dict[f"{ns}.Eval.x"] = 10.0
        disc_dict[f"{ns}.Eval.Coupling.Disc1.a"] = 1.0
        disc_dict[f"{ns}.Eval.Coupling.Disc1.b"] = 100.0
        disc_dict[f"{ns}.y"] = 4.0
        disc_dict[f"{ns}.Eval.Coupling.Disc1.indicator"] = 53.0

        uc_cls = Study(run_usecase=True)
        uc_cls.load_data(from_input_dict=disc_dict)

        dm = uc_cls.execution_engine.dm
        scenario_namespace_computed = dm.get_value(f"{ns}.tornado_chart_analysis.scenario_variations")
        scenarios = ["reference_scenario"]
        scenarios.extend([f"scenario_{i}" for i in range(1, 5)])
        scenario_namespace = {
            "scenario_name": scenarios,
            "Coupling.Disc1.a": [0.0, -10.0, 10.0, 0.0, 0.0],
            "Coupling.Disc1.b": [0.0, 0.0, 0.0, -10.0, 10.0],
        }
        self.assertDictEqual(scenario_namespace_computed.to_dict(orient="list"), scenario_namespace)

        samples_df_computed = dm.get_value(f"{ns}.Eval.samples_df")
        samples_df = {
            "selected_scenario": [True] * 5,
            "scenario_name": scenarios,
            "Coupling.Disc1.a": [1.0, 0.9, 1.1, 1.0, 1.0],
            "Coupling.Disc1.b": [100.0, 100.0, 100.0, 100.0 * (1 - 10.0 / 100.0), 100.0 * (1 + 10.0 / 100.0)],
        }
        self.assertDictEqual(samples_df_computed.to_dict(orient="list"), samples_df)

        uc_cls.run()
        variations_output_df_computed = dm.get_value(f"{ns}.tornado_chart_analysis.y_dict_variations")
        self.assertIsNotNone(variations_output_df_computed)

        from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory

        post_processing_factory = PostProcessingFactory()
        charts = post_processing_factory.get_post_processing_by_namespace(
            uc_cls.execution_engine, f"{uc_cls.study_name}.tornado_chart_analysis", None, as_json=False
        )

        self.assertIsNotNone(charts)
        self.assertTrue(len(charts) > 0)
