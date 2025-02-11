'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2024/06/10 Copyright 2023 Capgemini.

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

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tools.folder_operations import rmtree_safe


class TestUncertaintyQuantification(unittest.TestCase):
    """UncertaintyQuantification test class."""

    def setUp(self):
        """Initialize third data needed for testing."""
        self.repo = 'sostrades_core.sos_processes.test'
        self.mod_path = 'sostrades_core.sos_wrapping.analysis_discs.uncertainty_quantification'
        self.proc_name = 'test_uncertainty_quantification_analysis'
        self.name = 'Test'
        self.uncertainty_quantification = 'UncertaintyQuantification'

        self.ee = ExecutionEngine(self.name)
        self.factory = self.ee.factory
        self.dir_to_del = []

    def tearDown(self):
        """Test teardown."""
        for directory in self.dir_to_del:
            if Path(directory).is_dir():
                rmtree_safe(directory)

    def test_01_uncertainty_quantification(self):
        """Test uncertainty quantification."""
        builder = self.factory.get_builder_from_process(self.repo, self.proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        ns_dict = {
            'ns_sample_generator': f'{self.name}.{self.uncertainty_quantification}',
            'ns_evaluator': f'{self.name}.{self.uncertainty_quantification}',
            'ns_uncertainty_quantification': f'{self.name}.UncertaintyQuantification',
        }

        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.configure()

        self.ee.display_treeview_nodes()

        self.data_dir = Path(__file__).parents[1] / f"sos_processes/test/{self.proc_name}/data"

        self.samples_dataframe = pd.read_csv(self.data_dir / "samples_df.csv")

        # fixes a particular state of the random generator algorithm thanks to
        # the seed sample_size
        generator = np.random.default_rng(seed=42)
        var1 = generator.uniform(-1, 1, size=len(self.samples_dataframe))
        var2 = generator.uniform(-1, 1, size=len(self.samples_dataframe))

        out1 = list(pd.Series(var1 + var2) * 100000)
        out2 = list(pd.Series(var1 * var2) * 100000)
        out3 = list(pd.Series(np.square(var1) + np.square(var2)) * 100000)

        self.data_df = pd.DataFrame({
            'scenario_name': self.samples_dataframe['scenario_name'],
            'output1': out1,
            'output2': out2,
            'output3': out3,
        })

        input_selection = {
            'selected_input': [True, True, True],
            'shortest_name': ['COC', 'RC', 'NRC'],
            'full_name': ['COC', 'RC', 'NRC'],
        }

        output_selection = {
            'selected_output': [True, True, True],
            'shortest_name': ['output1', 'output2', 'output3'],
            'full_name': ['output1', 'output2', 'output3'],
        }

        dspace = pd.DataFrame({
            'variable': ['COC', 'RC', 'NRC'],
            'shortest_name': ['COC', 'RC', 'NRC'],
            'lower_bnd': [85.0, 80.0, 80.0],
            'upper_bnd': [105.0, 120.0, 120.0],
            'nb_points': [10, 10, 10],
            'full_name': ['COC', 'RC', 'NRC'],
        })

        private_values = {
            f'{self.name}.{self.uncertainty_quantification}.samples_inputs_df': self.samples_dataframe,
            f'{self.name}.{self.uncertainty_quantification}.samples_outputs_df': self.data_df,
            f'{self.name}.{self.uncertainty_quantification}.design_space': dspace,
            f'{self.name}.{self.uncertainty_quantification}.eval_inputs': pd.DataFrame(input_selection),
            f'{self.name}.{self.uncertainty_quantification}.gather_outputs': pd.DataFrame(output_selection),
        }

        self.ee.load_study_from_input_dict(private_values)
        self.ee.configure()
        self.ee.execute()

        uncertainty_quanti_disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.uncertainty_quantification}'
        )[0]

        uncertainty_quanti_disc_output = uncertainty_quanti_disc.get_sosdisc_outputs()
        uncertainty_quanti_disc_output['output_interpolated_values_df']

        chart_filter = uncertainty_quanti_disc.get_chart_filter_list()
        uncertainty_quanti_disc.get_post_processing_list(chart_filter)
        # for graph in graph_list:
        #    graph.to_plotly().show()

    def test_02_uncertainty_quantification_from_cartesian_product(self):
        """In this test we prove the ability to couple a grid search and an uq."""
        proc_name = 'test_mono_driver_with_uq'
        repo_name = self.repo + ".tests_driver_eval.mono"
        builder = self.factory.get_builder_from_process(repo_name, proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.load_study_from_input_dict({})

        disc1_name = 'Disc1'
        ns = f'{self.ee.study_name}'
        dspace_dict = {
            'variable': [f'subprocess.{disc1_name}.a', 'x'],
            'shortest_name': [f'subprocess.{disc1_name}.a', 'x'],
            'full_name': [f'subprocess.{disc1_name}.a', 'x'],
            'nb_points': [10, 10],
            'lower_bnd': [0.0, 0.0],
            'upper_bnd': [10.0, 10.0],
        }
        dspace = pd.DataFrame(dspace_dict)

        output_selection_obj_y1_y2 = {
            'selected_output': [True, True, False],
            'full_name': [f'subprocess.{disc1_name}.indicator', 'z', 'y'],
        }
        output_selection_obj_y1_y2 = pd.DataFrame(output_selection_obj_y1_y2)

        disc_dict = {}
        # DoE inputs
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = 'cartesian_product'

        a_list = np.linspace(0, 10, 2).tolist()
        x_list = np.linspace(0, 10, 2).tolist()

        eval_inputs_cp = pd.DataFrame({
            'selected_input': [True, False, True, False],
            'full_name': [
                f'subprocess.{disc1_name}.a',
                f'subprocess.{disc1_name}.b',
                'x',
                'subprocess.Disc2.power',
            ],
            'list_of_values': [a_list, [], x_list, []],
        })
        disc_dict[f'{self.ee.study_name}.Eval.with_sample_generator'] = True
        disc_dict[f'{self.ee.study_name}.SampleGenerator.eval_inputs'] = eval_inputs_cp
        disc_dict[f'{ns}.SampleGenerator.design_space'] = dspace
        disc_dict[f'{ns}.Eval.gather_outputs'] = output_selection_obj_y1_y2

        disc_dict[f'{ns}.Eval.x'] = 10.0
        disc_dict[f'{ns}.Eval.subprocess.{disc1_name}.a'] = 5.0
        disc_dict[f'{ns}.Eval.subprocess.{disc1_name}.b'] = 2.0
        disc_dict[f'{ns}.Eval.subprocess.Disc2.constant'] = math.pi
        disc_dict[f'{ns}.Eval.subprocess.Disc2.power'] = 2

        self.ee.load_study_from_input_dict(disc_dict)

        self.ee.execute()

    def test_03_uncertainty_quantification_with_arrays_in_inputs(self):
        """This tests evaluates the capacity to perform uncertainty quantification when some inputs are arrays."""
        builder = self.factory.get_builder_from_process(self.repo, self.proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        ns_dict = {
            'ns_sample_generator': f'{self.name}.{self.uncertainty_quantification}',
            'ns_evaluator': f'{self.name}.{self.uncertainty_quantification}',
            'ns_uncertainty_quantification': f'{self.name}.UncertaintyQuantification',
        }

        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.configure()

        self.ee.display_treeview_nodes()

        self.data_dir = Path(__file__).parents[1] / f"sos_processes/test/{self.proc_name}/data"
        self.samples_dataframe = pd.read_csv(self.data_dir / "samples_df.csv")

        # builds a sample dataframe with regular arrays samples
        x_range = np.arange(50, 150, 15)
        y_range = np.arange(40, 80, 8)
        z_range = np.arange(60, 120, 15)
        x, y, z = np.meshgrid(x_range, y_range, z_range)
        triplets = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        samples_dataframe = pd.concat([self.samples_dataframe[:-1]] * len(triplets))
        samples_dataframe = pd.concat([samples_dataframe, self.samples_dataframe.iloc[-1:]])
        array_var_column = []
        for triplet in triplets:
            array_var_column += [triplet] * len(self.samples_dataframe[:-1])
        array_var_column.append(10.0)
        samples_dataframe['input_array'] = array_var_column
        samples_dataframe['scenario_name'] = [f'scenario_{i}' for i in range(len(samples_dataframe) - 1)] + [
            'reference_scenario'
        ]
        self.samples_dataframe = samples_dataframe

        # fixes a particular state of the random generator algorithm thanks to
        # the seed sample_size
        generator = np.random.default_rng(seed=42)
        var1 = generator.uniform(-1, 1, size=len(self.samples_dataframe))
        var2 = generator.uniform(-1, 1, size=len(self.samples_dataframe))
        out1 = list(pd.Series(var1 + var2) * 100000)
        out2 = list(pd.Series(var1 * var2) * 100000)
        out3 = list(pd.Series(np.square(var1) + np.square(var2)) * 100000)

        self.data_df = pd.DataFrame({
            'scenario_name': self.samples_dataframe['scenario_name'],
            'output1': out1,
            'output2': out2,
            'output3': out3,
        })

        input_selection = {
            'selected_input': [True, True, True, True],
            'shortest_name': ['COC', 'RC', 'NRC', 'input_array'],
            'full_name': ['COC', 'RC', 'NRC', 'input_array'],
        }

        output_selection = {
            'selected_output': [True, True, True],
            'shortest_name': ['output1', 'output2', 'output3'],
            'full_name': ['output1', 'output2', 'output3'],
        }

        dspace = pd.DataFrame({
            'shortest_name': ['COC', 'RC', 'NRC', 'input_array'],
            'lower_bnd': [85.0, 80.0, 80.0, np.array([50.0, 40.0, 60])],
            'upper_bnd': [105.0, 120.0, 120.0, np.array([150.0, 80.0, 120.0])],
            'nb_points': [10, 10, 10, 10],
            'full_name': ['COC', 'RC', 'NRC', 'input_array'],
            'variable': ['COC', 'RC', 'NRC', 'input_array'],
        })

        private_values = {
            f'{self.name}.{self.uncertainty_quantification}.samples_inputs_df': self.samples_dataframe,
            f'{self.name}.{self.uncertainty_quantification}.samples_outputs_df': self.data_df,
            f'{self.name}.{self.uncertainty_quantification}.design_space': dspace,
            f'{self.name}.{self.uncertainty_quantification}.eval_inputs': pd.DataFrame(input_selection),
            f'{self.name}.{self.uncertainty_quantification}.gather_outputs': pd.DataFrame(output_selection),
        }

        self.ee.load_study_from_input_dict(private_values)
        self.ee.configure()
        self.ee.execute()

        uncertainty_quanti_disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.uncertainty_quantification}'
        )[0]

        uncertainty_quanti_disc_output = uncertainty_quanti_disc.get_sosdisc_outputs()
        uncertainty_quanti_disc_output['output_interpolated_values_df']

        chart_filter = uncertainty_quanti_disc.get_chart_filter_list()
        uncertainty_quanti_disc.get_post_processing_list(chart_filter)
        """
        for graph in graph_list:
            graph.to_plotly().show()
        """

    def test_04_uncertainty_quantification_with_arrays_in_input_and_outputs(self):
        """
        This tests evaluates the capacity to perform uncertainty quantification when some inputs are arrays and
        some outputs are arrays.
        """
        builder = self.factory.get_builder_from_process(self.repo, self.proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        ns_dict = {
            'ns_sample_generator': f'{self.name}.{self.uncertainty_quantification}',
            'ns_evaluator': f'{self.name}.{self.uncertainty_quantification}',
            'ns_uncertainty_quantification': f'{self.name}.UncertaintyQuantification',
        }

        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.configure()

        self.ee.display_treeview_nodes()

        self.data_dir = Path(__file__).parents[1] / f"sos_processes/test/{self.proc_name}/data"
        self.samples_dataframe = pd.read_csv(self.data_dir / "samples_df.csv")

        # builds a sample dataframe with regular inputs arrays
        x_range = np.arange(50, 150, 15)
        y_range = np.arange(40, 80, 8)
        z_range = np.arange(60, 120, 15)
        x, y, z = np.meshgrid(x_range, y_range, z_range)
        triplets = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        samples_dataframe = pd.concat([self.samples_dataframe[:-1]] * len(triplets))
        samples_dataframe = pd.concat([samples_dataframe, self.samples_dataframe.iloc[-1:]])
        array_var_column = []
        for triplet in triplets:
            array_var_column += [triplet] * len(self.samples_dataframe[:-1])
        array_var_column.append(10.0)
        samples_dataframe['input_array'] = array_var_column
        samples_dataframe['scenario_name'] = [f'scenario_{i}' for i in range(len(samples_dataframe) - 1)] + [
            'reference_scenario'
        ]
        self.samples_dataframe = samples_dataframe

        # fixes a particular state of the random generator algorithm thanks to
        # the seed sample_size
        generator = np.random.default_rng(seed=42)
        var1 = generator.uniform(-1, 1, size=len(self.samples_dataframe))
        var2 = generator.uniform(-1, 1, size=len(self.samples_dataframe))
        out1 = list(pd.Series(var1 + var2) * 100000)

        # set outputs to be both floats and arrays
        out1 = list(pd.Series(var1 + var2) * 100000)
        out_array = list(
            np.array([
                (var1 * var2) * 100_000,
                (var1**2 + var2**2) * 100_000,
                (var1**4 - var2**2) * 100_000,
                (-(var1**2) - var2**2) * 100_000,
            ]).T
        )

        self.data_df = pd.DataFrame({
            'scenario_name': self.samples_dataframe['scenario_name'],
            'output1': out1,
            'output_array': out_array,
        })

        input_selection = {
            'selected_input': [True, True, True, True],
            'shortest_name': ['COC', 'RC', 'NRC', 'input_array'],
            'full_name': ['COC', 'RC', 'NRC', 'input_array'],
        }

        output_selection = {
            'selected_output': [True, True],
            'shortest_name': ['output1', 'output_array'],
            'full_name': ['output1', 'output_array'],
        }

        dspace = pd.DataFrame({
            'shortest_name': ['COC', 'RC', 'NRC', 'input_array'],
            'lower_bnd': [85.0, 80.0, 80.0, np.array([50.0, 40.0, 60])],
            'upper_bnd': [105.0, 120.0, 120.0, np.array([150.0, 80.0, 120.0])],
            'nb_points': [10, 10, 10, 10],
            'full_name': ['COC', 'RC', 'NRC', 'input_array'],
            'variable': ['COC', 'RC', 'NRC', 'input_array'],
        })

        private_values = {
            f'{self.name}.{self.uncertainty_quantification}.samples_inputs_df': self.samples_dataframe,
            f'{self.name}.{self.uncertainty_quantification}.samples_outputs_df': self.data_df,
            f'{self.name}.{self.uncertainty_quantification}.design_space': dspace,
            f'{self.name}.{self.uncertainty_quantification}.eval_inputs': pd.DataFrame(input_selection),
            f'{self.name}.{self.uncertainty_quantification}.gather_outputs': pd.DataFrame(output_selection),
        }

        self.ee.load_study_from_input_dict(private_values)
        self.ee.configure()
        self.ee.execute()

        uncertainty_quanti_disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.uncertainty_quantification}'
        )[0]

        uncertainty_quanti_disc_output = uncertainty_quanti_disc.get_sosdisc_outputs()
        uncertainty_quanti_disc_output['output_interpolated_values_df']

        chart_filter = uncertainty_quanti_disc.get_chart_filter_list()
        uncertainty_quanti_disc.get_post_processing_list(chart_filter)
        # for graph in graph_list:
        #     graph.to_plotly().show()
