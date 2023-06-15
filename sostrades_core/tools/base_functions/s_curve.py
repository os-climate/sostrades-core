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
S-Curve function to be used for efficiency improvement for example
'''

import numpy as np


def s_curve(x, coeff=0.1, power=1, x0=0.0, y_min=0.0, y_max=1.0):
    '''
    S-Curve or Sigmoid function  between 0 and 1

    coeff is the slope of the S-curve more the coeff is high, more the slope is steep
    power is the curve of the S-curve near limit values, more the power is high more the curve is pronounced

    x0 is the middle of the S-curve if power=1, if power != 1 y[x0] = (1/2)**power

    y_min is the Scurve minimum
    y_max is the Scurve maximum 
    '''
    y = (y_max - y_min) / (1.0 + np.exp(-coeff * (x - x0)))**power + y_min

    return y
