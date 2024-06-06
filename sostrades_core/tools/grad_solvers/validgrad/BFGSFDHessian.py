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
# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding: iso-8859-1 -*-
import numpy


class BFGSFDHessian():
    
    def __init__(self,scheme,df_pointer):
        self.__scheme=scheme
        self.__dfpointer=df_pointer
        
    def get_scheme(self):
        return self.__scheme
    
    def iterate_bfgs_mat(self,H,xk,xkp1,grad_k,grad_kp1):
        sk=numpy.atleast_2d(xkp1-xk).T#numpy.atleast_2d
        yk=numpy.atleast_2d(grad_kp1-grad_k).T
#        print "sk"+str(sk)
#        print "yk"+str(yk)
#        print "numpy.dot(yk.T,sk)"+str(numpy.dot(yk.T,sk))
#        print "numpy.dot(numpy.dot(sk.T,H),sk)"+str(numpy.dot(numpy.dot(sk.T,H),sk))
        H+=numpy.dot(yk,yk.T)/numpy.dot(yk.T,sk)-numpy.dot(numpy.dot(H,sk),numpy.dot(sk.T,H))/numpy.dot(numpy.dot(sk.T,H),sk)
#        print "BFGS approximation = \n"+str(H)
#        d2x=numpy.atleast_2d(xkp1).T
#        if numpy.dot(numpy.dot(d2x.T,H),d2x)<0.:
#            print "H is not positive definite !"
        return H
    
    def hess_f(self,x):
        """
        Compute hessian BFGS approximation
        """
        self.__scheme.set_x(x)
        self.__scheme.generate_samples()
        H=numpy.eye(len(x))
        grad_k=self.__dfpointer(x)
        xk=x
        for x in self.__scheme.get_samples():
            grad_kp1=self.__dfpointer(x)
            xkp1=x
            H=self.iterate_bfgs_mat(H,xk,xkp1,grad_k,grad_kp1)
            grad_k=grad_kp1
            xk=xkp1
        return H

    def vect_hess_f(self,x):
        """
        Vectorized hessian computation
        """
        self.__scheme.set_x(x)
        self.__scheme.generate_samples()
        
        grad_k=self.__dfpointer(x)
        nb_f=numpy.shape(grad_k)[0]
        H_list=[]
        for i in range(nb_f):
            H_list.append(numpy.eye(len(x)))
        xk=x
        for x_up in self.__scheme.get_samples():
            grad_kp1=self.__dfpointer(x_up)
            xkp1=x_up
            for i,H in enumerate(H_list):
                #print "update with newgrad"+str(grad_kp1[i,:])+' vs '+str(grad_k[i,:])
                H_list[i]=self.iterate_bfgs_mat(H,xk,xkp1,grad_k[i,:],grad_kp1[i,:])
            grad_k=grad_kp1
            xk=xkp1
        return H_list
