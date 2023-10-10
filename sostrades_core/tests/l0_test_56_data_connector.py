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
from os import remove
from sostrades_core.tools.tree.serializer import DataSerializer
from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from time import sleep
from pathlib import Path

from sostrades_core.execution_engine.data_connector.mock_connector import MockConnector
from sostrades_core.execution_engine.data_connector.data_connector_factory import ConnectorFactory
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.sos_processes.test.test_disc1_data_connector_dremio.usecase import Study
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class TestMetadataDiscipline(SoSWrapp):
    """
    Discipline to test desc_in metadata for connector
    """
    dremio_path = '"test_request"'

    data_connection_dict = {ConnectorFactory.CONNECTOR_TYPE: MockConnector.NAME,
                            'hostname': 'test_hostname',
                            'connector_request': dremio_path}

    DESC_IN = {
        'deliveries': {'type': 'float', 'value': None, 'namespace': 'ns_market_deliveries', 'editable': False,
                       ProxyDiscipline.CONNECTOR_DATA: data_connection_dict}
    }

    def run(self):
        self.deliveries = self.get_sosdisc_inputs(
            ['deliveries'], in_dict=True)
        print(self.deliveries)


class TestWriteDataDiscipline(SoSWrapp):
    """
    Discipline to test writting data with connector
    """
    dremio_path = '"test_request"'

    data_connection_dict = {ConnectorFactory.CONNECTOR_TYPE: MockConnector.NAME,
                            'hostname': 'test_hostname',
                            'connector_request': dremio_path,
                            'connector_mode': 'write'}

    DESC_OUT = {
        'deliveries_df': {'type': 'float', 'value': None, 'namespace': 'ns_market_deliveries', 'editable': False,
                          ProxyDiscipline.CONNECTOR_DATA: data_connection_dict}
    }

    def run(self):
        connector_info = self.DESC_OUT['deliveries_df'][ProxyDiscipline.CONNECTOR_DATA]
        # no need to call "write_data" method because it is done in
        # fill_output_with_connecotr in sos_discipline


class TestDataConnector(unittest.TestCase):
    """
    Data connector test class
    """

    def setUp(self):
        '''
        Initialize single disc process
        '''
        self.file_to_del = join(dirname(__file__), 'data', 'dm.pkl')
        self.tearDown()
        self.name = 'EETests'
        self.model_name = 'test_'
        self.ee = ExecutionEngine(self.name)

        self.dremio_path = '"test_request"'
        self.data_connection_dict = {ConnectorFactory.CONNECTOR_TYPE: MockConnector.NAME,
                                     'hostname': 'test_hostname',
                                     'connector_request': self.dremio_path}
        self.test_connector = MockConnector()

        '''
        Initialize process for 2 disc in coupling
        '''
        self.base_path = 'sostrades_core.sos_wrapping.test_discs'
        self.dic_status = {'CONFIGURE': False,
                           'RUNNING': False,
                           'DONE': False}
        self.exec_eng = ExecutionEngine(self.name)

        ns_dict = {'ns_ac': self.name}
        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        mod_path = f'{self.base_path}.disc3_data_connector.Disc3'
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_path)

        mod_path = f'{self.base_path}.disc2_data_connector.Disc2_data_connector'
        disc2_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc2_data_connector', mod_path)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.exec_eng.configure()

        self.process = self.exec_eng.root_process

        '''
        Initialize process for 1 disc, data connector variable as output
        '''
        self.base_path2 = 'sostrades_core.sos_wrapping.test_discs'
        self.dic_status2 = {'CONFIGURE': False,
                            'RUNNING': False,
                            'DONE': False}
        self.exec_eng2 = ExecutionEngine(self.name)

        ns_dict2 = {'ns_ac': self.name}
        self.exec_eng2.ns_manager.add_ns_def(ns_dict2)

        mod_path2 = f'{self.base_path}.disc1_data_connector.Disc1'
        disc1_builder2 = self.exec_eng2.factory.get_builder_from_module(
            'Disc1', mod_path2)

        self.exec_eng2.factory.set_builders_to_coupling_builder(
            [disc1_builder2])

        self.exec_eng2.configure()

        self.process2 = self.exec_eng2.root_process

        self.error_message_assert_database = "Value from datamanager is different from the used database, check the configuration file of database access and/or used database/dm"
    def tearDown(self):
        if Path(self.file_to_del).is_file():
            remove(self.file_to_del)
            sleep(0.5)

    def test_01_data_connector_factory(self):
        """
        test the data_connector_factory
        """
        data = ConnectorFactory.use_data_connector(
            self.data_connection_dict)

        self.assertEqual(data, 42.0)

    def test_02_meta_data_desc_in(self):
        """
        Case where variable with data_connector is an input and does not come from another model.
        Data connector is needed
        """
        ns_dict = {'ns_market_deliveries': self.name}
        self.ee.ns_manager.add_ns_def(ns_dict)
        mod_path = 'sostrades_core.tests.l0_test_56_data_connector.TestMetadataDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)
        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.load_study_from_input_dict({})
        self.ee.display_treeview_nodes()
        self.ee.execute()
        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]

        self.assertEqual(disc.mdo_discipline_wrapp.wrapper.deliveries['deliveries'], 42.0)

    def test_03_input_data_connector_with_coupling(self):
        """
        Case where variable with data connector comes from the output of an other model
        Data connector should not be used
        """

        # modify DM
        values_dict = {}
        values_dict['EETests.Disc3.a'] = 10.
        values_dict['EETests.Disc3.b'] = 5.
        values_dict['EETests.x'] = 2.
        values_dict['EETests.Disc2_data_connector.constant'] = 4.
        values_dict['EETests.Disc2_data_connector.power'] = 2

        self.exec_eng.load_study_from_input_dict(values_dict)
        res = self.exec_eng.execute()
        y = self.exec_eng.dm.get_value('EETests.y')
        self.assertEqual(y, 25.0)

    def test_04_output_data_connector(self):
        """
        Case where variable with data connector is an output.
        Data connector need to be used +read if the meta_data data_connector is saved in pickles
        """

        # modify DM
        values_dict = {}
        values_dict['EETests.Disc1.a'] = 10.
        values_dict['EETests.Disc1.b'] = 5.
        values_dict['EETests.x'] = 2.

        self.exec_eng2.load_study_from_input_dict(values_dict)
        res = self.exec_eng2.execute()

        y = self.exec_eng2.dm.get_value('EETests.y')
        self.assertEqual(y, 42.0)

        folder_path = join(dirname(__file__), 'data')
        serializer = DataSerializer()
        serializer.put_dict_from_study(
            folder_path, DirectLoadDump(), self.exec_eng2.dm.convert_data_dict_with_full_name())
        sleep(0.1)
        ref_dm_df = serializer.get_dict_from_study(
            folder_path, DirectLoadDump())
        self.assertTrue('EETests.y' in ref_dm_df.keys(), 'no y in file')
        if 'EETests.y' in ref_dm_df.keys():
            data_to_read = ref_dm_df['EETests.y']
            print(data_to_read)
            self.assertTrue(
                ProxyDiscipline.CONNECTOR_DATA in data_to_read.keys(), 'no metadata in file')

    def test_05_write_data(self):
        """
        Test to write data with connector
        """
        ns_dict = {'ns_market_deliveries': self.name}
        self.ee.ns_manager.add_ns_def(ns_dict)
        mod_path = 'sostrades_core.tests.l0_test_56_data_connector.TestWriteDataDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)
        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.load_study_from_input_dict({})
        self.ee.display_treeview_nodes()
        self.ee.execute()
        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        deliveries_df = disc.get_sosdisc_outputs('deliveries_df')

        self.assertTrue(deliveries_df is not None)

    def _test_06_process_data_connector_dremio(self):
        """
        Test data connector dremio process
        """
        exec_eng = ExecutionEngine(self.name)
        builder_process = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_disc1_data_connector_dremio')

        exec_eng.factory.set_builders_to_coupling_builder(builder_process)

        exec_eng.configure()

        study_dremio = Study()
        study_dremio.study_name = self.name
        dict_values_list = study_dremio.setup_usecase()

        dict_values = {}
        for dict_val in dict_values_list:
            dict_values.update(dict_val)

        exec_eng.load_study_from_input_dict(dict_values)

        exec_eng.execute()

    def test_07_mongodb(self):
        '''
        Test MongoDB data connector for local and shared namespaces
        '''
        study_name = 'usecase'
        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory
        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_disc1_two_ns_db'
        builder_proc = factory.get_builder_from_process(repo=self.repo,
                                                        mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder_proc)

        exec_eng.configure()

        disc_dict = {f'{study_name}.x': 1.,
                     f'{study_name}.a': 1.,
                     f'{study_name}.Disc1.b': 1.}
        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.configure()
        exec_eng.execute()
        dm = exec_eng.dm
        x_dm = dm.get_value(f'{study_name}.x')
        b_dm = dm.get_value(f'{study_name}.Disc1.b')
        a_dm = dm.get_value(f'{study_name}.a')

        a_db = 2
        b_db = 3
        x_db = 7

        # in process, ns_a (for variable x) is related to database, ns_b (for variable a) is not. Local namespace is linked to database
        # assert that value in dm is from database for variables x and b but not for a

        assert x_dm == x_db and b_dm == b_db and a_dm != a_db , self.error_message_assert_database

    def test_08_trino_two_db(self):
        '''
        Test Trino with two different databases data connector for local and shared namespaces
        '''

        # initialize classes, import process        
        study_name = 'usecase'
        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory
        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_disc1_two_ns_db_trino'
        builder_proc = factory.get_builder_from_process(repo=self.repo,
                                                        mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder_proc)

        exec_eng.configure()

        # set values for input variables
        disc_dict = {f'{study_name}.x': 2.,
                     f'{study_name}.a': 2.,
                     f'{study_name}.Disc1.b': 2.}
        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.configure()

        # execute usecase
        exec_eng.execute()
        # get dm and get values from dm 
        dm = exec_eng.dm
        x_dm = dm.get_value(f'{study_name}.x')
        b_dm = dm.get_value(f'{study_name}.Disc1.b')
        a_dm = dm.get_value(f'{study_name}.a')


        # in process, ns_a (for variable x) is related to database, ns_b (for variable a) is not. Local namespace is linked to database
        # assert that value in dm is from database for variables x and b but not for a. Check of values is done after the execution because values from database are
        # defined during "prepare_execution" step of execution_engine.

        # values from the databases for test
        a_db_1 = 3.2
        a_db_2 = 0.
        b_db = 0.
        x_db = 1.

        # assert value in dm is from database disca for x variable and discb for b
        assert x_dm == x_db and b_dm == b_db and a_dm != a_db_1 and a_dm != a_db_2 , self.error_message_assert_database



    def _test_10_data_commons_extraction(self):
        '''
        Test the extraction of tables in data commons with mysql request
        The test is commented because it is necessary to configure the credentials.env and the CREDENTIAL_DOTENV_DIR variable
        '''
        import os
        import pathlib
        from dotenv import load_dotenv
        import trino
        import pandas as pd
        from sqlalchemy.engine import create_engine

        dotenv_dir = os.environ.get("CREDENTIAL_DOTENV_DIR", os.environ.get("PWD", "/opt/app-root/src"))
        dotenv_path = pathlib.Path(dotenv_dir) / "credentials.env"
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path, override=True)

        sqlstring = 'trino://{user}@{host}:{port}/'.format(
            user=os.environ['TRINO_USER'],
            host=os.environ['TRINO_HOST'],
            port=os.environ['TRINO_PORT']
        )

        sqlargs = {
            'auth': trino.auth.JWTAuthentication(os.environ['TRINO_PASSWD']),
            'http_scheme': 'https'
        }
        engine = create_engine(sqlstring, connect_args=sqlargs)
        connection = engine.connect()

        qres = engine.execute('show catalogs')
        catalog_list = qres.fetchall()
        print('\n list of catalogs', catalog_list)
        for catalog in catalog_list:
            schemas_list = []
            try:
                qres = engine.execute(f'show schemas in {catalog[0]}')
                schemas_list = qres.fetchall()
                print(f'\n list of schemas in catalog {catalog[0]}', schemas_list)
            except:
                print(f'the catalog {catalog[0]} cannot be open')

            if schemas_list != []:
                for schema in schemas_list:
                    qres = engine.execute(f'show tables from {catalog[0]}.{schema[0]}')
                    tables_list = qres.fetchall()
                    print(f'\n list of tables in schema {schema[0]}', tables_list)
        catalog_name = 'osc_datacommons_dev'
        schema_name = 'demo_dv'
        # NOTE: BigQuery table names may include characters such as '-' (dash) that are not standard SQL identifier chars,
        # so you can see we enclose the table name in double quotes so trino will not complain
        table_name = 'template_cumulative_emissions'
        example_table_name = '.'.join([catalog_name, schema_name, table_name])

        qres = engine.execute(f"describe {example_table_name}")
        description = qres.fetchall()
        print(description)
        # qres = engine.execute('select * from ' + example_table_name)
        # example_table = qres.fetchall()
        pd.set_option('display.max_columns', None)
        # df = pd.DataFrame(example_table)
        # print(df)

        df = pd.read_sql(f"""
        select * from {example_table_name}""", engine)
        print(df)


if '__main__' == __name__:
    testcls = TestDataConnector()
    # testcls.setUp()
    testcls.test_07_mongodb()
