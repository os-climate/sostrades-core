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
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from os.path import join, dirname
from pathlib import Path
from os import remove
from time import sleep
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from sostrades_core.execution_engine.execution_engine import ExecutionEngine

LOC_DIRNAME = dirname(__file__)


class TestCouplingVariables(unittest.TestCase):
    """
    Coupling variables test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'EETests'
        self.repo = 'sostrades_core.sos_processes.test'
        self.file_to_del = join(dirname(__file__), 'MyCase.csv')

    def tearDown(self):
        if Path(self.file_to_del).is_file():
            remove(self.file_to_del)
            sleep(0.5)

    def test_01_export_sos_export_couplings(self):
        '''
        check export_couplings method in sos_coupling (not recursive)
        '''
        namespace = 'MyCase'
        ee = ExecutionEngine(namespace)
        ee.select_root_process(self.repo, 'test_disc1_disc2_coupling')
        ee.configure()
        # check treeview structure
        exp_tv_list = ['Nodes representation for Treeview MyCase',
                       '|_ MyCase',
                       '\t|_ Disc1',
                       '\t|_ Disc2']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == ee.display_treeview_nodes()
        # -- setup inputs
        dm = ee.dm
        values_dict = {}
        values_dict[f'{namespace}.Disc2.constant'] = -10.
        values_dict[f'{namespace}.Disc2.power'] = -10.
        values_dict[f'{namespace}.Disc1.a'] = 10.
        values_dict[f'{namespace}.Disc1.b'] = 20.
        values_dict[f'{namespace}.Disc1.indicator'] = 10.
        values_dict[f'{namespace}.x'] = 3.

        dm.set_values_from_dict(values_dict)

        ee.configure()
        rp = ee.root_process
        
        # gather couplings data
        df = rp.export_couplings()
        # compare df of couplings to ref
        data = [['MyCase.Disc1', 'MyCase.Disc2', 'MyCase.y']]
        header = ["disc_1", "disc_2", "var_name"]
        df_ref = DataFrame(data, columns=header)
        assert_frame_equal(df, df_ref, "wrong dataframe of couplings")

        # "test_13_export_couplings",
        f_name = join(LOC_DIRNAME, f"{rp.get_disc_full_name()}.csv")
        rp.export_couplings(in_csv=True, f_name=f_name)

    def test_02_checktype_unit_mismatch(self):
        '''
        check_var_data_mismatch method in sos_coupling (not recursive)
        '''
        namespace = 'MyCase'
        ee = ExecutionEngine(namespace)
        ee.select_root_process(self.repo, 'test_disc1_disc2_coupling')
        ee.configure()
        # check treeview structure
        exp_tv_list = ['Nodes representation for Treeview MyCase',
                       '|_ MyCase',
                       '\t|_ Disc1',
                       '\t|_ Disc2']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == ee.display_treeview_nodes()
        # -- setup inputs
        ee.dm
        values_dict = {}
        values_dict[f'{namespace}.Disc2.constant'] = -10.
        values_dict[f'{namespace}.Disc2.power'] = -10.
        values_dict[f'{namespace}.Disc1.a'] = 10.
        values_dict[f'{namespace}.Disc1.b'] = 20.
        values_dict[f'{namespace}.Disc1.indicator'] = 10.
        values_dict[f'{namespace}.x'] = 3.

        ee.load_study_from_input_dict(values_dict)

        rp = ee.root_process
        ee.prepare_execution()
        
        # gather couplings data
        rp.check_var_data_mismatch()


if '__main__' == __name__:
    cls = TestCouplingVariables()
    cls.setUp()
    cls.test_02_checktype_unit_mismatch()
