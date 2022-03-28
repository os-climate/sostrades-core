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
# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8 -*-
'''
Implementation of Sellar Disciplines (Sellar, 1996)
Adapted from GEMSEO examples
'''

from cmath import exp, sqrt
from numpy import array, atleast_2d, complex128, ones, zeros
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
import pandas as pd


#from sos_trades_core.sos_wrapping.test_discs.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.core import SobieskiProblem

class SobieskiMission(SoSDiscipline):
    """ Sobieski range wrapper using the Breguet formula.
    """
    _maturity = 'Fake'
    DESC_IN = {
                'y_14': {'type': 'array','default': array([50606.97417114, 7306.20262124]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_24': {'type': 'array','default': array([4.15006276]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_34': {'type': 'array','default': array([1.10754577]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'z': {'type': 'array','default': array([0.05, 45000.0, 1.6, 5.5, 55.0, 1000.0]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
               }

    DESC_OUT = {'y_4': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'}}

    def run(self):
        """ computes
        """
        y_14, y_24, y_34, z = self.get_sosdisc_inputs(['y_14', 'y_24', 'y_34', 'z'])
        DTYPE_DOUBLE = "float64"
        dtype = DTYPE_DOUBLE
        sobieski_problem = SobieskiProblem(dtype=dtype)
        y_4 = sobieski_problem.blackbox_mission(z,y_14, y_24, y_34)
        out = {'y_4': array(y_4)}
        self.store_sos_outputs_values(out)



    def compute_sos_jacobian(self):
        """

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """

        y_14, y_24, y_34, z = self.get_sosdisc_inputs(['y_14', 'y_24', 'y_34', 'z'])
        DTYPE_DOUBLE = "float64"
        dtype = DTYPE_DOUBLE
        sobieski_problem = SobieskiProblem(dtype=dtype)
        gemseo_jac_dict = sobieski_problem.derive_blackbox_mission(z, y_14, y_24, y_34)
        #We need to convert 'x_shared' variable of gemseo model into 'z'
        gemseo_jac_pd = pd.DataFrame(gemseo_jac_dict)
        sos_jac_pd = gemseo_jac_pd.rename(index = {'x_shared':'z'})
        sos_jac_dict = sos_jac_pd.to_dict()
        self.jac = sos_jac_dict

class SobieskiStructure(SoSDiscipline):
    """ Sobieski mass estimation wrapper.
    """
    _maturity = 'Fake'
    DESC_IN = {
                'x_1': {'type': 'array','default':array([0.25, 1.0]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_21': {'type': 'array','default':array([50606.9741711]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_31': {'type': 'array','default':array([6354.32430691]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'z': {'type': 'array','default':array([0.05, 45000.0, 1.6, 5.5, 55.0, 1000.0]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
               }

    DESC_OUT = {
                'y_1': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_11': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_14': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'g_1': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_12': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'}
                }

    def run(self):
        """ computes
        """
        x_1, y_21, y_31, z = self.get_sosdisc_inputs(['x_1', 'y_21', 'y_31', 'z'])
        DTYPE_DOUBLE = "float64"
        dtype = DTYPE_DOUBLE
        sobieski_problem = SobieskiProblem(dtype=dtype)
        y_1, y_11, y_12, y_14, g_1 = sobieski_problem.blackbox_structure(z, y_21, y_31, x_1)
        out = {
                'y_1': array(y_1),
                'y_11': array(y_11),
                'y_12': array(y_12),
                'y_14': array(y_14),
                'g_1': array(g_1)
                }
                
        self.store_sos_outputs_values(out)

    def compute_sos_jacobian(self):
        """

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """

        x_1, y_21, y_31, z = self.get_sosdisc_inputs(['x_1', 'y_21', 'y_31', 'z'])
        DTYPE_DOUBLE = "float64"
        dtype = DTYPE_DOUBLE
        sobieski_problem = SobieskiProblem(dtype=dtype)
        gemseo_jac_dict = sobieski_problem.derive_blackbox_structure(z, y_21, y_31, x_1)
        #We need to convert 'x_shared' variable of gemseo model into 'z'
        gemseo_jac_pd = pd.DataFrame(gemseo_jac_dict)
        sos_jac_pd = gemseo_jac_pd.rename(index = {'x_shared':'z'})
        sos_jac_dict = sos_jac_pd.to_dict()
        self.jac = sos_jac_dict

class SobieskiAerodynamics(SoSDiscipline):
    """ Sobieski aerodynamic discipline wrapper.
    """
    _maturity = 'Fake'
    DESC_IN = {
                'x_2': {'type': 'array','default':array([1.0]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_12': {'type': 'array','default': array([50606.9742, 0.95]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_32': {'type': 'array','default': array([0.50279625]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'z': {'type': 'array','default':array([0.05, 45000.0, 1.6, 5.5, 55.0, 1000.0]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
               }

    DESC_OUT = {
                'y_2': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_21': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_23': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_24': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'g_2': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'}
                }

    def run(self):
        """ computes
        """
        x_2, y_12, y_32, z = self.get_sosdisc_inputs(['x_2', 'y_12', 'y_32', 'z'])
        DTYPE_DOUBLE = "float64"
        dtype = DTYPE_DOUBLE
        sobieski_problem = SobieskiProblem(dtype=dtype)
        y_2, y_21, y_23, y_24, g_2 = sobieski_problem.blackbox_aerodynamics(z, y_12, y_32, x_2)
        out = {
                'y_2': array(y_2),
                'y_21': array(y_21),
                'y_23': array(y_23),
                'y_24': array(y_24),
                'g_2': array(g_2)
                }
                
        self.store_sos_outputs_values(out)
        
    def compute_sos_jacobian(self):
        """

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """

        x_2, y_12, y_32, z = self.get_sosdisc_inputs(['x_2', 'y_12', 'y_32', 'z'])
        DTYPE_DOUBLE = "float64"
        dtype = DTYPE_DOUBLE
        sobieski_problem = SobieskiProblem(dtype=dtype)
        gemseo_jac_dict = sobieski_problem.derive_blackbox_aerodynamics(z, y_12, y_32, x_2)
        #We need to convert 'x_shared' variable of gemseo model into 'z'
        gemseo_jac_pd = pd.DataFrame(gemseo_jac_dict)
        sos_jac_pd = gemseo_jac_pd.rename(index = {'x_shared':'z'})
        sos_jac_dict = sos_jac_pd.to_dict()
        self.jac = sos_jac_dict
        
class SobieskiPropulsion(SoSDiscipline):
    """ Sobieski propulsion propulsion wrapper.
    """
    _maturity = 'Fake'
    DESC_IN = {
                'x_3': {'type': 'array','default':array([0.5]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_23': {'type': 'array','default': array([12562.01206488]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'z': {'type': 'array','default':array([0.05, 45000.0, 1.6, 5.5, 55.0, 1000.0]), 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
               }

    DESC_OUT = {
                'y_3': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_34': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_31': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'y_32': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'},
                'g_3': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSobieski'}
                }

    def run(self):
        """ computes
        """
        x_3, y_23, z = self.get_sosdisc_inputs(['x_3', 'y_23', 'z'])
        DTYPE_DOUBLE = "float64"
        dtype = DTYPE_DOUBLE
        sobieski_problem = SobieskiProblem(dtype=dtype)
        y_3, y_34, y_31, y_32, g_3 = sobieski_problem.blackbox_propulsion(z, y_23, x_3)
        out = {
                'y_3': array(y_3),
                'y_34': array(y_34),
                'y_31': array(y_31),
                'y_32': array(y_32),
                'g_3': array(g_3)
                }
                
        self.store_sos_outputs_values(out)
        
    def compute_sos_jacobian(self):
        """

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """

        x_3, y_23, z = self.get_sosdisc_inputs(['x_3', 'y_23', 'z'])
        DTYPE_DOUBLE = "float64"
        dtype = DTYPE_DOUBLE
        sobieski_problem = SobieskiProblem(dtype=dtype)
        gemseo_jac_dict = sobieski_problem.derive_blackbox_propulsion(z, y_23, x_3)
        gemseo_jac_pd = pd.DataFrame(gemseo_jac_dict)
        sos_jac_pd = gemseo_jac_pd.rename(index = {'x_shared':'z'})
        sos_jac_dict = sos_jac_pd.to_dict()
        self.jac = sos_jac_dict



if __name__ == '__main__':
    disc_id = 'coupling_sob_disc'
    namespace = "test"
    ee = ExecutionEngine("test")
