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
from sos_trades_core.sos_processes.test import test_uncertainty_quantification_analysis
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import random

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.sos_wrapping.analysis_discs.uncertainty_quantification import UncertaintyQuantification
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from os.path import join, dirname


class TestUncertaintyQuantification(unittest.TestCase):
    """
    UncertaintyQuantification test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.repo = 'sos_trades_core.sos_processes.test'
        self.mod_path = 'sos_trades_core.sos_wrapping.analysis_discs.uncertainty_quantification'
        self.proc_name = 'test_uncertainty_quantification_analysis'
        self.name = 'Test'
        self.uncertainty_quantification = 'UncertaintyQuantification'

        self.ee = ExecutionEngine(self.name)
        self.factory = self.ee.factory

    def test_01_uncertainty_quantification(self):

        repo = 'sos_trades_core.sos_processes.test'
        mod_path = 'sos_trades_core.sos_wrapping.analysis_discs.uncertainty_quantification'

        builder = self.factory.get_builder_from_process(
            self.repo, self.proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        ns_dict = {'ns_ac': f'{self.name}', 'ns_public': f'{self.name}',
                   'ns_energy': f'{self.name}',
                   'ns_coc_ac': f'{self.name}',
                   'ns_dmc_ac': f'{self.name}',
                   'ns_data_ac': f'{self.name}',
                   'ns_coc': f'{self.name}',
                   'ns_uncertainty_quantification': f'{self.name}.UncertaintyQuantification'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.configure()

        self.ee.display_treeview_nodes()

        self.data_dir = join(dirname(dirname(
            __file__)), 'sos_processes', 'test', self.proc_name, 'data')

        self.samples_dataframe = pd.read_csv(
            join(self.data_dir, 'samples_df.csv'))

        Var1 = np.random.uniform(-1, 1, size=len(self.samples_dataframe))
        Var2 = np.random.uniform(-1, 1, size=len(self.samples_dataframe))

        out1 = list(pd.Series(Var1 + Var2) * 100000)
        out2 = list(pd.Series(Var1 * Var2) * 100000)
        out3 = list(pd.Series(np.square(Var1) + np.square(Var2)) * 100000)

        self.data_df = pd.DataFrame(
            {'scenario': self.samples_dataframe['scenario'], 'output1': out1, 'output2': out2, 'output3': out3})

        private_values = {
            f'{self.name}.{self.uncertainty_quantification}.samples_df': self.samples_dataframe,
            f'{self.name}.{self.uncertainty_quantification}.data_df': self.data_df,
            # f'{self.name}.{self.uncertainty_quantification}.input_distribution_parameters_df': self.input_distribution_parameters_df,
            # f'{self.name}.{self.uncertainty_quantification}.data_details_df':
            # self.data_details_df,
        }

        self.ee.load_study_from_input_dict(private_values)
        self.ee.configure()
        self.ee.execute()

        uncertainty_quanti_disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.uncertainty_quantification}')[0]

        uncertainty_quanti_disc_output = uncertainty_quanti_disc.get_sosdisc_outputs()
        out_df = uncertainty_quanti_disc_output['output_interpolated_values_df']

        filter = uncertainty_quanti_disc.get_chart_filter_list()
        graph_list = uncertainty_quanti_disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #     graph.to_plotly().show()


if '__main__' == __name__:
    cls = TestUncertaintyQuantification()
    cls.setUp()
    unittest.main()
