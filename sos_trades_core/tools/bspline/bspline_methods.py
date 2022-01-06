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
from sos_trades_core.tools.bspline.bspline import BSpline


def bspline_method(ctrl_pts, length):
    '''
    Method to use Bspline on a set of ctrl points over a length
    '''
    list_t = np.linspace(0.0, 1.0, length)

    bspline = BSpline(n_poles=len(ctrl_pts))
    bspline.set_ctrl_pts(ctrl_pts)

    t_list = bspline.eval_list_t(list_t)

    return t_list
