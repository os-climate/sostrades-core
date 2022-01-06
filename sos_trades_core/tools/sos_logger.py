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
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from os import remove
from pathlib import Path
from os.path import abspath
from sqlite3 import connect
from re import findall, escape
from time import strftime, localtime, sleep

from logging.handlers import RotatingFileHandler
from logging import CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET
from logging import Logger, getLogger, Formatter, StreamHandler, FileHandler, basicConfig

MSG_SEP = '\t'
MSG_FMT = MSG_SEP.join(['%(asctime)s.%(msecs)03d ',
                        '%(name)-20s',
                        '%(levelname)-10s',
                        '%(message)s'])
TIME_FMT = '%Y-%m-%d %H:%M:%S'
SOS_FMT = {'format': MSG_FMT, 'datefmt': TIME_FMT}


class SoSLogging(Logger):
    '''
    Initialize the logger object
    First, checks if log_file entry is just a filename then create a directory
    Then, instantiate Logger class 
    Finally returns the logger object
    '''
    CRITICAL = CRITICAL
    FATAL = FATAL
    ERROR = ERROR
    WARNING = WARNING
    WARN = WARN
    INFO = INFO
    DEBUG = DEBUG
    NOTSET = NOTSET
    DEFAULT_LEVEL = INFO
    log_file = None

    def __init__(self, name,
                 master=False, handler_mode='stream',
                 filename=None,
                 level=INFO):
        '''
        differentiate root logger (master=True) to sub loggers 
        in order to configure logger then sub loggers (by inheritance) once
        '''
        self.master = master
        basicConfig(level=INFO)
        self.logger = getLogger(name)
        if master:
            hdl_msg = None
            new_hdlr = self.new_handler_after_cleaning(filename)
            if 'file' in handler_mode.lower():
                if new_hdlr:
                    handler = SoSFileHandler(filename)
                    print(
                        'FileHandler mode chosen, logs will be written into file', filename)
                    hdl_msg = handler.hdl_msg
            elif 'sqlite' in handler_mode.lower():
                if new_hdlr:
                    handler = SoSSQLiteHandler(db_filename=filename)
                    print(
                        'SQLiteHandler mode chosen, logs will be written into dB', filename)
                    hdl_msg = handler.hdl_msg
            else:  # stream mode
                handler = StreamHandler()
                print('StreamHandler mode chosen, logs will be written into console')

            self.logger.setLevel(level)

            if new_hdlr:
                formatter = Formatter(SOS_FMT['format'],
                                      SOS_FMT['datefmt'])
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

            options_msg = ''
            if self.logger.level < self.INFO:
                options_msg = f' with handler mode {handler_mode}'
                log_file_msg = f' and logs are written into {self.log_file}' if self.log_file else ''
                print(f'Master logger initialized{options_msg}{log_file_msg}')
            self.logger.info(f'Master logger initialized{options_msg}')
            if hdl_msg is not None:
                self.logger.debug(hdl_msg)
            # end if master
        self.log_file = filename

    def new_handler_after_cleaning(self, filename):
        ''' 
        loop on handlers, if new filename is the same, keep this handler
        and do not add a new one
        else remove handler
        '''
        new_hdlr = True
        for hdlr in self.logger.handlers:
            hdl_fn = getattr(hdlr, 'baseFilename', 'no file for logging')
            if filename != hdl_fn:
                hdlr.close()
                self.logger.removeHandler(hdlr)
            else:
                new_hdlr = False
        return new_hdlr

    def get_all_log_files(self, root=False):
        if root:
            handlers = getLogger().handlers
        else:
            handlers = self.logger.handlers
        log_files = {}
        for a_hdlr in handlers:
            hdlr_type = type(a_hdlr).__name__
            log_files[hdlr_type] = a_hdlr.baseFilename if 'stream' in hdlr_type else None
        return log_files

    def get_log_file(self):
        return self.log_file

    def stop(self):
        for hdlr in self.logger.handlers:
            hdlr.close()
            self.logger.removeHandler(hdlr)


class SoSFileHandler(RotatingFileHandler):
    def __init__(self, filename=None, handler_mode='FileHandler',
                 maxBytes=None, maxLines=None, backupCount=None):
        self.hdl_msg = None
        if filename is None:
            raise IOError(
                f'file path shall be given for mode {handler_mode} with argument')
        filename = abspath(filename)
        if Path(filename).is_file():
            self.hdl_msg = f'logger file already exists {filename}. It will be appended'
        # assuming one line is about 100 bytes
        maxBytes = maxBytes or 1e7
        if maxLines is not None:
            maxBytes = min(maxBytes, maxLines * 100)
        backupCount = backupCount or 5
        super().__init__(filename=filename,
                         maxBytes=maxBytes, backupCount=backupCount,
                         encoding='utf-8', delay=False)


class SoSSQLiteHandler(FileHandler):
    """
    Logging handler for SQLite inherited from FileHandler
    """

    initial_sql = """CREATE TABLE IF NOT EXISTS log(
                        Created text,
                        Name text,
                        LogLevel int,
                        LogLevelName text,    
                        Message text,
                        Args text,
                        Module text,
                        FuncName text,
                        LineNo int,
                        Exception text,
                        Process int,
                        Thread text,
                        ThreadName text
                   )"""

    insertion_sql = """INSERT INTO log(
                        Created,
                        Name,
                        LogLevel,
                        LogLevelName,
                        Message,
                        Args,
                        Module,
                        FuncName,
                        LineNo,
                        Exception,
                        Process,
                        Thread,
                        ThreadName
                   )
                   VALUES (
                        '%(dbtime)s',
                        '%(name)-20s',
                         %(levelno)d,
                        '%(levelname)-10s',
                        '%(msg)s',
                        '',
                        '%(module)s',
                        '%(funcName)s',
                         %(lineno)d,
                        '%(exc_text)s',
                         %(process)d,
                        '%(thread)s',
                        '%(threadName)s'
                   );
                   """
    sql_fields = findall(escape('%(') + "(.*)" + escape(')'), insertion_sql)

    def __init__(self, db_filename):
        # When initialized, existing database will be deleted
        self.hdl_msg = None
        if Path(db_filename).is_file():
            self.hdl_msg = f'logger file already exists {db_filename}. It will be erased'
            try:
                conn = connect(db_filename)
                if conn:
                    conn.close()
                    sleep(0.1)
                remove(db_filename)
                sleep(0.1)
            except OSError as os_er:
                os_msg = str(os_er).replace("'", '"')
                self.hdl_msg = f'logger file already exists {db_filename} but it cannot be erased OSError: {os_msg}'
        super().__init__(filename=db_filename,
                         mode='w', encoding='utf-8', delay=False)
        # Create table if needed:
        conn = connect(self.baseFilename)
        conn.execute(SoSSQLiteHandler.initial_sql)
        conn.commit()

    def formatDBTime(self, record):
        record.dbtime = strftime(TIME_FMT, localtime(record.created))

    def emit(self, record):
        sql = ''
        conn = None
        try:
            # Use default formatting:
            self.format(record)
            # Set the database time up:
            self.formatDBTime(record)
            if record.exc_info:
                record.exc_text = Formatter().formatException(record.exc_info)
            else:
                record.exc_text = ""

            # Escape special character in string values
            for k in self.sql_fields:
                v = getattr(record, k)
                if isinstance(v, str):
                    setattr(record, k, v.replace("'", "''"))
                elif v.__class__.__name__ == 'Exception':
                    setattr(record, k, str(v).replace("'", '"'))

            # Instanciate msg with argument format
            record.msg = record.msg % record.args

            # Reset args to avoir manipulate tuple in database
            record.args = ''

            if '%' in record.msg:
                record.msg = record.msg % record.args

            # Insert log record:
            sql = SoSSQLiteHandler.insertion_sql % record.__dict__

            conn = connect(self.baseFilename)
            conn.execute(sql)
            conn.commit()
        except Exception as error:
            # standard output for the error in order to have the logs of it
            # (use logger for log an error wil result in an infinite loop)
            print(error, sql)
        finally:
            if conn:
                conn.close()
