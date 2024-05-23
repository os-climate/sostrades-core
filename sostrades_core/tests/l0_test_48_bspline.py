'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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
import unittest

import numpy as np
from scipy.interpolate import BSpline as bspline_sp

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tools.bspline.bspline import BSpline


class TestBSpline(unittest.TestCase):
    """
    Class to test BSpline tools
    """

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_01_execute(self):
        bsp = BSpline(degree=3, n_poles=8)
        ctrl = np.array([-0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7])
        bsp.set_ctrl_pts(ctrl)
        x = np.linspace(0.0, 1.0, 80)

        eval1, _ = bsp.eval_list_t(x)

        bsp_scipy = bspline_sp(bsp.knots, ctrl, 3)
        result_scipy = []
        for elem in x:
            result_scipy.append(float(bsp_scipy(elem)))
        np.testing.assert_almost_equal(eval1.tolist(), result_scipy)
