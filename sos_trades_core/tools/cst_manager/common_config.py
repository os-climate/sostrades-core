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
import numpy as np

gravity = 9.80665    # Earth-surface gravitational acceleration (m/s**2)
rad_to_deg = 180. / np.pi
deg_to_rad = np.pi / 180.
ft2m = 0.3048     # feet to meter
mtft = 3.2808399  # meter to feet
kt2m_s = 0.5144444
Nm2m = 1852
###
# "Awesome" breaking model
mu_r = 0.02       # Rolling friction coefficient
mu_b = 0.6        # Breaking friction coefficient

nr_solver_conf = {}
nr_solver_conf['eps'] = 10**-5
nr_solver_conf['stop_residual'] = 10**-7
nr_solver_conf['max_ite'] = 20
nr_solver_conf['relax_factor'] = 0.95
