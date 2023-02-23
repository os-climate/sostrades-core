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


class ConstraintObject:
    """
    Class to instantiate a constraint for mission/vehicle design
    (if GEMS this class should not be needed anymore)
    """

    def __init__(self, values, weights=None):
        """
        Constructor for the Constraintobject class
        """
        if isinstance(values, np.ndarray):
            self.values = values
        elif isinstance(values, list):
            self.values = np.array(values)
        else:
            self.values = np.array([values])

        if weights is None:
            self.weights = np.ones_like(self.values)
        elif isinstance(weights, np.ndarray):
            self.weights = weights
        elif isinstance(weights, list):
            self.weights = np.array(weights)
        else:
            self.weights = np.array([weights])
