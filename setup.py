'''
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License. 
'''
# coding: utf-8

from setuptools import setup, find_packages
import platform

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

reqs_list = [
    'nose2==0.9.1',
    'pycryptodome==3.9.8',
    'plotly==5.3.0',
    'pandas==1.3.0',
    'numpy==1.20.3',
    'cma==3.1.0',
    'ortools==9.0.9048',
    'PyYAML==5.1.2',
    'dremio_client==0.15.1',
    'pyarrow==6.0.1',
    'pydoe2==1.3.0',
    'nlopt==2.6.2',
    'scipy==1.7.1',
    'gemseo==3.2.1',
    'openturns==1.18',
    'chaospy==4.3.7',
    'sympy==1.4',
    'trino',
    'cvxpy==1.1.18',
]

if platform.system() != 'Windows':
    reqs_list += 'petsc==3.12.3'
    reqs_list += 'petsc4py==3.12.0'

setup(
    name='sos-trades-core',
    version='0.1.0',
    description='Core of System of System Trades',
    long_description=readme,
    author='Airbus SAS',
    url='https://idas661.eu.airbus.corp/sostrades/sos_trades_core.git',
    license=license,
    packages=find_packages(exclude=('tests')),
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=reqs_list,
)
