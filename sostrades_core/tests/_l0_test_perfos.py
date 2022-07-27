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
import cProfile
import pstats
import unittest
from io import StringIO
from os.path import dirname, join
from pathlib import Path
from shutil import rmtree
from time import sleep

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.sos_processes.test.test_disc1.usecase import Study


class TestScatter(unittest.TestCase):
    """
    SoSDiscipline test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'

    def tearDown(self):

        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

    def test_01_perfos_test_disc1_process(self):

        self.name = 'usecase'
        self.ee = ExecutionEngine(self.name)
        repo = 'sostrades_core.sos_processes'
        builder = self.ee.factory.get_builder_from_process(
            repo, 'test.test_disc1')

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        usecase = Study(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        profil = cProfile.Profile()
        profil.enable()
        
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        
        profil.disable()
        result = StringIO()

        ps = pstats.Stats(profil, stream=result)
        ps.sort_stats('cumulative')
        ps.print_stats(1000)
        result = result.getvalue()
        # chop the string into a csv-like buffer
        result = 'ncalls' + result.split('ncalls')[-1]
        result = '\n'.join([','.join(line.rstrip().split(None, 5))
                            for line in result.split('\n')])
        

        with open(join(dirname(__file__), 'test_disc1_perfos.csv'), 'w+') as f:
            # f = open(result.rsplit('.')[0] + '.csv', 'w')
            f.write(result)
            f.close()

        lines = result.split('\n')
        total_time = float(lines[1].split(',')[3])
        print('total_time : ', total_time)


if '__main__' == __name__:
    cls = TestScatter()
    cls.test_01_perfos_test_disc1_process()
