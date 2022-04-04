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
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from numpy import atleast_2d, array
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine



# numerical example from 
# Lundell, A., Kronqvist, J. Polyhedral approximation strategies for nonconvex mixed-integer nonlinear programming in SHOT. J Glob Optim (2021).
# https://link.springer.com/article/10.1007/s10898-021-01006-1

# numerical example from 
# https://blogs.sas.com/content/iml/2017/01/18/milp-sas.html
# expected sol is "x1=5, x2=3.1"

class DiscMixedOpt(SoSDiscipline):
    __maturity = 'Fake'
    DESC_IN = {
        'x1': {'type': 'array'},
        'x2': {'type': 'array'}
    }

    DESC_OUT = {
        'obj': {'type': 'array'},
        'constr': {'type': 'array'}
    }

    def run(self):
        
        x1 = self.get_sosdisc_inputs('x1')[0]
        x2 = self.get_sosdisc_inputs('x2')[0]
        
        o = -(3*x1 +  5*x2)
        obj = array([o])

        c = []
        c.append( 3*x1 -  2*x2 - 10)
        c.append( 5*x1 + 10*x2 - 56)
        c.append(-4*x1 -  2*x2 + 7 )
        constr = array(c)
        dict_values = {'obj': obj, 'constr': constr}
        
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
        
    def compute_sos_jacobian(self):
#         x1, x2 = self.get_sosdisc_inputs(['x1', 'x2'])

        self.set_partial_derivative('obj', 'x1', atleast_2d(array([-3.])))
        self.set_partial_derivative('obj', 'x2', atleast_2d(array([-5.])))
        self.set_partial_derivative('constr', 'x1', atleast_2d(array([[ 3.],  [5.], [-4.]])))
        self.set_partial_derivative('constr', 'x2', atleast_2d(array([[-2.], [10.], [-2.]])))
    

# numerical example from 
# https://fdocuments.in/reader/full/solving-mixed-integer-nonlinear-programming-minlp-problems-mixed-integer-nonlinear

class DiscMixedOptUnfeas(SoSDiscipline):
    _maturity = 'Fake'
    DESC_IN = {
        'x': {'type': 'array'},
        'y': {'type': 'array'}
    }

    DESC_OUT = {
        'obj': {'type': 'array'},
        'constr': {'type': 'array'}
    }

    def run(self):
        
        x = self.get_sosdisc_inputs('x')
        y = self.get_sosdisc_inputs('y')
        
        obj = 0.5 * x + y
        constr = (x-1.)**2 + y**2 - 3.
        
        dict_values = {'obj': obj, 'constr': constr}
        
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
        
    def compute_sos_jacobian(self):

        x, y = self.get_sosdisc_inputs(['x', 'y'])

        self.set_partial_derivative('obj', 'x', atleast_2d(array([0.5])))
        self.set_partial_derivative('obj', 'y', atleast_2d(array([1.])))
        self.set_partial_derivative('constr', 'x', atleast_2d(array([2.*(x-1.)])))
        self.set_partial_derivative('constr', 'y', atleast_2d(array([2.*y])))
        
if __name__ == "__main__":
    
    exec_eng = ExecutionEngine("test")
    factory = exec_eng.factory
    mod = 'sos_trades_core.sos_wrapping.test_discs.disc_mixed_opt.DiscMixedOpt'
    builder = factory.get_builder_from_module('DiscMixedOpt', mod)
    factory.set_builders_to_coupling_builder(builder)
    exec_eng.configure()
    
    # check at integer solution 1
    disc_dict = {}
    disc_dict['test.DiscMixedOpt.x1'] = 1.
    disc_dict['test.DiscMixedOpt.x2'] = 2.
    
    exec_eng.load_study_from_input_dict(disc_dict)
    
    d = exec_eng.root_process
    d.check_jacobian(derr_approx=SoSDiscipline.COMPLEX_STEP)
    
    # check at integer solution 5
    disc_dict = {}
    disc_dict['test.DiscMixedOpt.x1'] = 5
    disc_dict['test.DiscMixedOpt.x2'] = 3.
    
    exec_eng.load_study_from_input_dict(disc_dict)
    
    d = exec_eng.root_process
    d.check_jacobian(derr_approx=SoSDiscipline.COMPLEX_STEP)
