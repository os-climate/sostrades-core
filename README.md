
# SoSTrades_CORE


## Description
SoSTradesCore is the Python package containing the core of the execution engine of System of Systems Trades

## Packages installation
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

## Overview
This package is divided into 6 parts:

- execution_engine: contains all the methods and wrapped class of gemseo tools necessary to implement processes and studies in SoSTrades
- sos_processes: contains test processes built with disciplines from sos_wrapping
- sos_wrapping: contains test disciplines covering execution engine functionalities
- study_manager: contains generic class to implement studies
- tests: contains tests on execution_engine functionalities, based on sos_processes and sos_wrapping
- tools: contains generic tools of execution_engine
