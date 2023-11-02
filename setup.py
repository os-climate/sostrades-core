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

# Read the requirements.in file and extract the required dependencies
with open('requirements.in', 'r') as requirements_file:
    reqs_list = [line.strip() for line in requirements_file if line.strip()]

# platform-specific dependencies added conditionally
if platform.system() != 'Windows':
    reqs_list.append('petsc==3.12.3')
    reqs_list.append('petsc4py==3.12.0')

setup(
    name='sos-trades-core',
    version='0.1.0',
    description='Core of System of System Trades',
    long_description=readme,
    author='Airbus SAS',
    url='https://github.com/os-climate/sostrades-core.git',
    license=license,
    packages=find_packages(exclude=('tests')),
    include_package_data=True,
    python_requires='>=3.9',
    install_requires=reqs_list,
)
