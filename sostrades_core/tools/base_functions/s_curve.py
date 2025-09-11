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

from __future__ import annotations

from typing import Union

import numpy as np

"""S-Curve function for efficiency improvements and sigmoid calculations."""


def s_curve(x: Union[float, np.ndarray],
           coeff: float = 0.1,
           power: float = 1.0,
           x0: float = 0.0,
           y_min: float = 0.0,
           y_max: float = 1.0) -> Union[float, np.ndarray]:
    """
    Generate S-Curve or Sigmoid function between specified bounds.

    Args:
        x: Input value(s) for curve calculation.
        coeff: Slope coefficient - higher values create steeper slope.
        power: Curve power near limit values - higher values create more pronounced curves.
        x0: Middle point of S-curve (when power=1, y[x0] = (1/2)**power).
        y_min: Minimum value of the S-curve.
        y_max: Maximum value of the S-curve.

    Returns:
        Calculated S-curve value(s) between y_min and y_max.

    """
    y = (y_max - y_min) / (1.0 + np.exp(-coeff * (x - x0)))**power + y_min
    return y
