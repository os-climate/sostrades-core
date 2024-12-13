'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/07-2023/11/03 Copyright 2023 Capgemini

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

import logging
import os
from os.path import dirname, join

logging.basicConfig(level=logging.INFO)

# set-up the folder where GEMSEO will look-up for new wrapps (solvers, grammars etc)
logging.getLogger('gemseo').setLevel('DEBUG')
parent_dir = dirname(__file__)
GEMSEO_ADDON_DIR = "gemseo_addon"
EXEC_ENGINE = "execution_engine"

os.environ["GEMSEO_PATH"] = join(parent_dir, EXEC_ENGINE, GEMSEO_ADDON_DIR)
