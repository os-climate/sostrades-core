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
from .base_controller import BaseController

from numpy import zeros

class Variable(BaseController):
    """
    Variable Class
    """
    #--Class variables
    CLASS_MSG = 'PVariable'
    ERROR_MSG = 'ERROR '+CLASS_MSG+'.'
    GRAD = None
    
    #--constructor
    def __init__(self,PBCManager,ID,value=0.0, bounds=None, complex_mode=False):
        BaseController.__init__(self, PBCManager, ID, value=value, BCType='Variable', complex_mode=complex_mode)
        self.__bounds = None
        self.set_bounds(bounds)
        
    #--Private methods
    def __repr__(self):
        """
        display some information about the variable
        """
        info_string=BaseController.__repr__(self)
        info_string += '\n   Value           :%24.16e'%self.get_value()
        info_string += '\n   Gradient        : '+str(self.get_gradient())
        return info_string

    #--Methods
    def get_bounds(self):
        return self.__bounds
    
    def set_bounds(self, bounds):
        ERROR_MSG=self.ERROR_MSG+'set_bounds: '
        if bounds is not None:
            self.__bounds = bounds
            lbnd=self.__bounds[0]
            ubnd=self.__bounds[1]
            if lbnd >= ubnd:
                raise Exception(ERROR_MSG+' Lower bound greater or equal to upper bound!')
    
    def handle_dv_changes(self):
        if self.__class__.GRAD is None or self.get_ndv() != len(self.__class__.GRAD):
            self.__class__.GRAD= zeros(self.get_ndv())
            
        self.set_gradient(self.__class__.GRAD)
        