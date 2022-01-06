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
from os import makedirs, remove
from os.path import join, dirname, isfile
from shutil import rmtree
from pathlib import Path
from time import sleep
from logging import shutdown, getLogger
from sqlite3 import connect
from pandas import read_sql

from sos_trades_core.tools.sos_logger import SoSLogging, MSG_SEP


class TestSoSLogging(unittest.TestCase):
    """
    SoS logger test class
    """

    def setUp(self):
        self.test_dir = join(dirname(__file__), 'data', 'tmp')
        self.file_to_del = None
        self.dir_to_del = None
        self.logger_name = 'Test_SoS_Logger'
        self.ref_dir = join(dirname(__file__), 'data', 'ref_output')
#         l_n_r = '{0: <20}'.format(self.logger_name)
        l_n_r = '%-20s' % self.logger_name
        lvl_m = '%-10s'
        self.log_msg_ref_warn = [[l_n_r, lvl_m % 'INFO', 'Master logger initialized'],
                                 [l_n_r, lvl_m % 'INFO', '0/ info'],
                                 # [l_n_r, lvl_m % 'INFO', '█    | 1/5'],
                                 # [l_n_r, lvl_m % 'INFO', '██   | 2/5'],
                                 # [l_n_r, lvl_m % 'INFO', '███  | 3/5'],
                                 # [l_n_r, lvl_m % 'INFO', '████ | 4/5'],
                                 # [l_n_r, lvl_m % 'INFO', '█████| 5/5'],
                                 [l_n_r, lvl_m % 'WARNING', '1/ warning'],
                                 [l_n_r, lvl_m % 'ERROR', '2/ error'],
                                 [l_n_r, lvl_m % 'ERROR',
                                     "('3/ Exception raised returning a tuple', ('1234',))"],
                                 [l_n_r, lvl_m % 'CRITICAL', '4/- critical'],
                                 [l_n_r, lvl_m % 'INFO', 'test SoS logger ends.']]

    def tearDown(self):
        root_loggger = getLogger()
        shutdown()
        for hdlr in root_loggger.handlers:
            root_loggger.removeHandler(hdlr)
        if self.file_to_del:
            if Path(self.file_to_del).is_file():
                remove(self.file_to_del)
                sleep(0.5)
        if self.dir_to_del:
            if Path(self.dir_to_del).is_dir():
                rmtree(self.dir_to_del)
                sleep(0.5)

    def bar_message(self, n_iter):
        max_iter = 5
        N_BARS = 5
        bar_l = chr(0x2588) * n_iter
        full_bar = bar_l + ' ' * max(N_BARS - n_iter, 0)
        return f'|{full_bar}| {n_iter}/{max_iter}'

    def run_logging_test(self, a_logging):
        a_logger = a_logging.logger
        a_logger.debug('test SoS logger starts...')
        a_logger.info('0/ info')
        for i in range(5):
            a_logger.info(self.bar_message(i + 1))
        sleep(0.1)
        a_logger.warning('1/ %s' % 'warning')
        sleep(0.1)
        a_logger.error('2/ {}'.format('error'))
        sleep(0.1)
        try:
            raise Exception('3/ Exception raised returning a tuple', ('1234',))
        except Exception as exc:
            a_logger.exception(exc)
        sleep(0.1)
        lvl = 'critical'
        a_logger.critical(f'4/- {lvl}')
        sleep(0.1)
        a_logger.info('test SoS logger ends.')

    def test_01_stream_logger(self):
        print('_' * 10, 'test sos logger on stream handler mode...')
        if Path(self.test_dir).is_dir():
            rmtree(self.test_dir)
            sleep(0.5)
        a_logging = SoSLogging(self.logger_name, master=True)
        self.run_logging_test(a_logging)
        assert len(a_logging.logger.handlers) == 1
        assert type(a_logging.logger.handlers[0]).__name__ == 'StreamHandler'
        assert a_logging.log_file is None
        assert a_logging.get_log_file() is None
        a_logging.stop()

    def test_02_file_logger(self):
        print('_' * 10, 'test sos logger on file handler mode...')
        if Path(self.test_dir).is_dir():
            rmtree(self.test_dir)
            sleep(0.5)
        makedirs(self.test_dir)
        log_file_bn = join(self.test_dir, 'test_sos_logger.log')
        a_logging = SoSLogging(self.logger_name, master=True,
                               handler_mode='filehandler',
                               filename=log_file_bn, level=SoSLogging.INFO)
        self.run_logging_test(a_logging)
        log_file = a_logging.get_log_file()
        assert isfile(log_file)
        assert(a_logging.log_file == log_file)

        log_msg_out_all = []
        with open(log_file, 'r') as l_f:
            for a_l in l_f.readlines():
                # remove date and time from message
                m_o = a_l.replace('\n', '')
                log_msg_out_all.append(MSG_SEP.join(m_o.split(MSG_SEP)[1:]))
        log_msg_out = log_msg_out_all[:2]
        log_msg_out += log_msg_out_all[7:10]
        log_msg_out += log_msg_out_all[14:]
        log_msg_ref = [MSG_SEP.join(l) for l in self.log_msg_ref_warn]
        self.assertListEqual(log_msg_ref, log_msg_out)
        a_logging.stop()
        self.file_to_del = log_file

    def test_03_sqlite_logger(self):
        print('_' * 10, 'test sos logger on sqlite handler mode...')
        if Path(self.test_dir).is_dir():
            rmtree(self.test_dir)
            sleep(0.5)
        makedirs(self.test_dir)
        db_name = 'test_2_sqlite_logger_app.db'
        out_sql_db = join(self.test_dir, db_name)
        a_logging = SoSLogging(self.logger_name, master=True,
                               filename=out_sql_db,
                               handler_mode='SQlite', level=SoSLogging.INFO)
        self.run_logging_test(a_logging)
        a_logging.stop()

        out_con = connect(out_sql_db)
        out_df = read_sql(
            'SELECT Name, Loglevelname, Message FROM log;', out_con)
        i = 0
        log_msg_out = []
        for a_ar in list(out_df.values):
            if i < 2 or i > len(out_df.values) - 6:
                log_msg_out.append(list(a_ar))
            i += 1
        log_msg_ref = self.log_msg_ref_warn
        exc_msg = log_msg_ref[-3][2].replace("'", 'simplequote')
        exc_msg = exc_msg.replace('"', "'")
        log_msg_ref[-3][2] = exc_msg.replace("simplequote", '"')
        self.assertListEqual(log_msg_ref, log_msg_out)

        self.dir_to_del = self.test_dir
        a_logging.stop()


if __name__ == "__main__":
    unittest.main()
