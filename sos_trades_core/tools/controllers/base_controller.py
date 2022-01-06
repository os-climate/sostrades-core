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
from .controller import Controller
import numpy as np

class BaseController(Controller):
    """
    BaseController Class
    """
    #--Class Variables
    ERROR_MSG='ERROR BaseController.'
    CLASSTYPE='BC'
    USE_GRADIENT_ARRAYS=True
    
    #--Constructor
    def __init__(self, BCManager, Id, value=0.0, BCType='Generic', complex_mode=False):
        self.complex_mode = complex_mode
        self.__BCType   = BCType
        self.__value    = None    #Initilizing attributes before updating the gradient in PController.handle_dv_changes
        self.__gradient = None
        
        Controller.__init__(self,BCManager,self.CLASSTYPE,Id)
        self.set_value(value)
        
    #--Private methods
    def __repr__(self):
        info_string  =Controller.__repr__(self)
        info_string += '\n   BC Type         : '+self.get_BCType()
        return info_string

    #--Accessors
    def get_BCType(self):
        return self.__BCType
    
    def get_value(self):
        return self.__value

    def get_gradient(self):
        return self.__gradient
    
    #--Setters        
    def set_value(self,value,flag_updates=True):
        if value is not None:
            if self.complex_mode:
                val = np.complex128(value)
            else:
                val = np.float(value.real)
        self.__value = val
        if flag_updates:
            self.set_controllers_to_update()
        
    def update_value(self,value, flag_updates=True):
        if self.__value!=value:
            self.set_value(value,flag_updates=flag_updates)
            
    def set_and_check_deps(self):
        pass
        
    def set_gradient(self,gradient):
        self.__gradient = gradient
        self.set_controllers_to_update()
