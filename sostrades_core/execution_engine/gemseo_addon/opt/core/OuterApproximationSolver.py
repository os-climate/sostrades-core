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
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

from gemseo.algos.driver_lib import DriverLib
from gemseo.algos.opt.opt_lib import OptimizationLibrary

import logging
from numpy import array, append, int32, atleast_2d, concatenate
from copy import deepcopy
import cvxpy as cp
from pandas.core.frame import DataFrame
import pandas as pd

# TODO list : 
# * avoid re-building the NLP
# * look for a solution to get output dimensions for cases where len(outvars)>0
# * exclude unfeasible integer solutions
# * map the max iter termination criteria
# * add UB/LB values to main database history
# * add post-processing

LOGGER = logging.getLogger("OuterApproximation")

class OuterApproximationSolver(object):
    '''
    Implementation of Outer Approximation solver
    '''
    ETA = "eta"
    UPPER_BOUND = "UB"
    FULL_PROBLEM_DV_NAME = "x"
    MILP_DV_NAME_INT = FULL_PROBLEM_DV_NAME + '_int'
    MILP_DV_NAME_FLOAT = FULL_PROBLEM_DV_NAME + '_float'
    ALGO_OPTIONS_MILP = "algo_options_MILP"
    ALGO_OPTIONS_NLP = "algo_options_NLP"
    ALGO_NLP = "algo_NLP"
    NORMALIZE_DESIGN_SPACE_OPTION = DriverLib.NORMALIZE_DESIGN_SPACE_OPTION
    MAX_ITER = OptimizationLibrary.MAX_ITER
    F_TOL_ABS = OptimizationLibrary.F_TOL_ABS
    # tags for problem database
    UPPER_BOUND_CANDIDATES = UPPER_BOUND + "_history"
    UPPER_BOUNDS = UPPER_BOUND
    LOWER_BOUNDS = "LB"
    OA_ITER_NB = "oa_ite_nb"


    def __init__(self, problem):
        '''
        Constructor
        '''
        self.full_problem = problem
        if not problem.minimize_objective:
            msg = "Problem defined as an objective maximization instead of minimization"
            raise ValueError(msg)
        self.dual_problem = None
        self.primal_problem = None
        self.epsilon = 1e-3
        self.upper_bounds_candidates = []
        self.upper_bounds = []
        self.lower_bounds = []
        self.cont_solutions = []
        self.int_solutions = []
        self.x_solution_history = []
        self.ind_by_varname, self.size_by_varname = None, None
        self.opt_history = None
        self.iter_nb = None
    
    def set_options(self, **options):
        
        self.differentiation_method = self.full_problem.differentiation_method
        self.max_iter = options[self.MAX_ITER]
        self.ftol_abs = options[self.F_TOL_ABS]
        
        self.algo_options_MILP = options[self.ALGO_OPTIONS_MILP]
        self.algo_NLP = options[self.ALGO_NLP]
        self.algo_options_NLP = options[self.ALGO_OPTIONS_NLP]

    def init_solver(self):
        msg = "\n\n***\nOuterApproximation Initialization\n***"
        LOGGER.info(msg)
        
        # check the problem to avoid cases that are not handled by this algorithm implementation
        dspace = self.full_problem.design_space
        self._check_problem(dspace, self.full_problem)
        
        # get design variables indexes and size
        self.ind_by_varname = dspace.get_variables_indexes(dspace.variables_names)
        self.size_by_varname = dspace.variables_sizes
        
        # set indices corresponding to integer variables
        iv_ind, fv_ind = array([], dtype=int32), array([], dtype=int32)
        iv_names, fv_names = [], []
        
        for vname in dspace.variables_names:
            v_ind = dspace.get_variables_indexes([vname])
            if dspace.get_type(vname) == [DesignSpace.INTEGER.value]: # pylint: disable=E0602,E1101
                iv_ind = append(iv_ind, v_ind)
                iv_names.append(vname)
            else:
                fv_ind = append(fv_ind, v_ind)
                fv_names.append(vname)
                
        self.integer_indices = iv_ind
        self.int_varnames = iv_names
        self.float_varnames = fv_names
        
        # set indices corresponding to float variables
        fv_ind = array([], dtype=int32)
        for fv in dspace.variables_names:
            if fv in self.float_varnames:
                fv_ind = append(fv_ind, dspace.get_variables_indexes([fv]))
        self.float_indices = fv_ind
        
        # set initial integer solution
        x0 = dspace.get_current_x()
        self.x0_integer = x0[self.integer_indices]
        
        msg = "Initial guess of integer solution is "
        msg += str(self.x0_integer)
        LOGGER.info(msg)
        
    def _check_problem(self, dspace, problem):
        ''' performs checks to avoid cases not handled by this algorithm implementation
        - checks if one vectorized design variable
        - 
        '''
        # checks if a dv vector components have different types
        for v in dspace.variables_names:
            if len(dspace.get_type(v)) > 1:
                msg = 'The design variable <%s> has several types instead of one for all components.\n' %v
                msg += '(different types for each component of the variable is not handled for now)'
                raise ValueError(msg)
        
        # checks if problem functions have only one output
        if len(problem.objective.outvars) > 1:
            raise ValueError("Several outputs in MDOFunction is not allowed")
         
        for c in problem.constraints:
            if len(c.outvars) > 1:
                raise ValueError("Several outputs in MDOFunction is not allowed")
                
    
    def _get_integer_variables_indices(self, dspace):
        ''' returns integer variables indices in xvect defined by the design space
        '''
        return self._get_x_indices_by_type(dspace, 
                                           DesignSpace.INTEGER.value) # pylint: disable=E0602,E1101
        
    def _get_float_variables_indices(self, dspace):
        ''' returns float variables indices in xvect defined by the design space
        '''
        return self._get_x_indices_by_type(dspace, 
                                           DesignSpace.FLOAT.value) # pylint: disable=E0602,E1101
    
    def _build_full_vect(self, float_vals, int_vals):
        ''' builds the global xvect with continuous and integer values
        '''
        fdspace = self.full_problem.design_space
        
        x = fdspace.get_current_x()
        
        if len(x) != len(self.float_indices) + len(self.integer_indices):
            msg = 'Sum of Integer and Float design variables '
            msg += 'components is not equal to the design space full size.'
            raise ValueError(msg)
        
        x[self.integer_indices] = int_vals
        x[self.float_indices] = float_vals
        
        return x
        
    
    #- primal problem definition
    
    def build_primal_problem(self):
        ''' build primal problem without hyperplanes
        (will be updated at other iterations)
        '''
        # design variables definition
        eta = cp.Variable(name=self.ETA)

        # objective definition
        obj = cp.Minimize(eta)
        
#         # constraints definition
#         # (could be handled in termination criteria)
#         ub = cp.Parameter(name=self.UPPER_BOUND)
#         constraints = [eta <= ub - self.epsilon] # eta <= U^(k) - eps
        
        # problem definition
        prob = cp.Problem(obj) #, constraints
        
        return prob
    
    def update_primal_problem(self, old_primal_pb, dual_pb, upper_bnd, x0):
        ''' update primal problem with new upper bound value U^{(k)}
        and supporting hyperplanes (linearizations of objecgives and constraints
        of the NLP(x_int)^{(k)} )
        '''
        full_dspace = self.full_problem.design_space
        x0_dict = full_dspace.array_to_dict(x0)
        #- gather eta design variable
        eta = old_primal_pb.var_dict[self.ETA]
        #- gather x design variable if exists (for iterations > 0)
        bounds_cst = []
        if len(old_primal_pb.var_dict) > 1:
            # if there the other design variables than eta have already been created
            # wee keep them and there is no need to re-build the bound constraints
            all_vars = old_primal_pb.var_dict
        else:
            # if not, x is created with associated bounds constraints
            # create float variables and associated bound constraints
            all_vars = {}
            for v in full_dspace.variables_names:
                # get the dimensions of the variable
                v_shape = full_dspace.get_current_x_dict()[v].shape
                # create the design variable as cvxpy object
                if v in self.float_varnames:
                    integer=False
                else:
                    integer=True
                dv = cp.Variable(v_shape, v, integer=integer)
                all_vars[v] = dv
                # build the constraint on the lower bounds
                lb = full_dspace.get_lower_bounds([v])
                bounds_cst.append(lb <= dv)
                # build the constraint on the upper bound
                ub = full_dspace.get_upper_bounds([v])
                bounds_cst.append(dv <= ub)
                
        #- setup of primal problem constraints :
        #- build dual pb objective linearization
        obj_jac = atleast_2d(self.full_problem.objective.jac(x0))
        data_size = deepcopy(self.size_by_varname)
        data_size.update({self.full_problem.objective.outvars[0]: obj_jac.shape[0]})
        obj_jac_dict = split_array_to_dict_of_arrays(obj_jac,
                                         data_size,
                                         self.full_problem.objective.outvars, 
                                         self.full_problem.design_space.variables_names,
                                         )
        # get objective function from main optimization problem
        obj_f = self.full_problem.objective.func(x0)
        # builds linearization of the objective wrt all design variables
        obj_lin = obj_f
        for v in self.full_problem.design_space.variables_names:
            x_v = all_vars[v]
            x0_v = x0_dict[v]
            dx = x_v - x0_v
            obj_lin += obj_jac_dict[self.full_problem.objective.outvars[0]][v] @ dx
        # set the objective hyperplane as constraint
        obj_lin = obj_lin <= eta
        
        #- build dual pb constraints linearization : c(x0) + dc/dx(x0) . (x - x0) <= 0
        cst_linearized = []
        # for each constraint function from main optimization problem
        # builds linearization of the constraint wrt all design variables
        for c in self.full_problem.constraints:
            # compute c(x0)
            cst_f = c.func(x0)
            # get dc/dx(x0)
            c_jac = atleast_2d(c.jac(x0))
            data_size = deepcopy(self.size_by_varname)
            data_size.update({c.outvars[0]: c_jac.shape[0]})
            c_jac_dict = split_array_to_dict_of_arrays(c_jac, 
                                            data_size,
                                             c.outvars, 
                                             self.full_problem.design_space.variables_names,
                                             )
            # compute c(x0) + dc/dx(x0) . (x - x0)
            cst_lin = cst_f
            for v in self.full_problem.design_space.variables_names:
                x_v = all_vars[v]
                x0_v = x0_dict[v]
                dx = x_v - x0_v
                cst_lin += c_jac_dict[c.outvars[0]][v] @ dx
                
            cst_linearized.append(cst_lin <= 0)
        
        # problem re-definition (cvxpy does not allow in-memory problem updates, excepted parameter values)
        hyperplanes = [obj_lin] + cst_linearized
        primal_pb = cp.Problem(old_primal_pb.objective, 
                               old_primal_pb.constraints + hyperplanes + bounds_cst)
        
        print("\n\n")
        print("HERE")
        print("obj " + str(old_primal_pb.objective))
        for c in old_primal_pb.constraints + hyperplanes + bounds_cst:
            print("c " + str(c))
        
## handled in the termination criteria
#         # update upper bound parameter value
#         ub = primal_pb.param_dict[self.UPPER_BOUND]
#         print("upper_bnd", upper_bnd)
#         ub.value = upper_bnd
        
        return primal_pb
    
    def solve_primal_problem(self, problem):
        ''' solve the primal problem
        '''
        msg = "\n\n######## MIP Solver \n\n"
        LOGGER.info(msg)
        
        # solver execution
        problem.solve(solver=cp.CBC, verbose=True)
        
        # updates the history of bounds and integer solutions according to the status
        if problem.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            LOGGER.info("Optimal value: %s" % problem.value)
            sol_int = array([])
            for dv in self.full_problem.design_space.variables_names:
                if dv in self.int_varnames:
                    val = problem.var_dict[dv].value
                    sol_int = append(sol_int, val)
            self.int_solutions.append(sol_int)
            self.lower_bounds.append(problem.value)
        else:
            sol_int = None
            self.int_solutions.append(None)
            self.lower_bounds.append(None)
        
        # display results
        for variable in problem.variables():
            LOGGER.info("Variable %s: value %s" % (variable.name(), variable.value))
        
        LOGGER.info("status:" + str(problem.status))
        LOGGER.info("optimal value " + str(problem.value))
        
        return sol_int

    #- dual problem definition
    
    def build_dual_problem(self, integer_values):
        ''' Build the dual problem
        '''
        # retrieve full problem
        full_pb = self.full_problem
        
        # original design space filtered without integer variables
        cont_vars = []
        for v in full_pb.design_space:
            if full_pb.design_space.get_type(v) == DesignSpace.FLOAT.value: # pylint: disable=E0602
                cont_vars.append(v)
        dspace = full_pb.design_space.filter(cont_vars, copy=True)
        
        input_dim = sum(full_pb.design_space.variables_sizes.values()) # use dspace.dimension
        
        # build restriction of original constraint functions
        LOGGER.info("integer_indices " + str(self.integer_indices))
        LOGGER.info("integer_values " + str(integer_values))
        LOGGER.info("input_dim "+ str(input_dim))
            
        cst_restricted = []
        for c in full_pb.constraints:
            # builds the restriction
            new_c_name = c.name + '_restricted'
            new_c = c.restrict(self.integer_indices, #frozen indexes
                               integer_values, #frozen values
                               input_dim,
                               name=new_c_name,
                               f_type=MDOFunction.TYPE_INEQ,
                               #expr=f"{f.name}(%s)",
                               args=None)
            # build the function with store in main problem database
            
            # append the function to the constraint list
            cst_restricted.append(new_c)
        
        # build restriction of original objective functions
        new_o_name = full_pb.objective.name + '_restricted'
        new_o = full_pb.objective.restrict(self.integer_indices, #frozen indexes
                                           integer_values, #frozen values
                                           input_dim,
                                           name=new_o_name,
                                           f_type=MDOFunction.TYPE_OBJ,
                                           #expr=f"{f.name}(%s)",
                                           args=None)
        
        # build dual problem
        pb = OptimizationProblem(dspace)
        # objective setup
        pb.objective = new_o
        # constraints setup
        for c in cst_restricted:
            pb.add_constraint(c, cstr_type=MDOFunction.TYPE_INEQ)
        pb.differentiation_method = self.differentiation_method  # either FINITE_DIFFERENCES or USER_GRAD
        
        # functions are preprocessed once here (before the call in DriverLib at execution)
        # so that from now nonprocessed_* functions are accessible (see update_nlp)
        options = self.algo_options_NLP
        pb.preprocess_functions(
            normalize=options.get(self.NORMALIZE_DESIGN_SPACE_OPTION, True),
            use_database=options.get(DriverLib.USE_DATABASE_OPTION, True),
            round_ints=options.get(DriverLib.ROUND_INTS_OPTION, True),
            eval_obs_jac=False,
        )
        
        return pb
    
     
    def update_dual_problem(self, nlp, integer_values):
        ''' Updates frozen values of NLP problem with those provided
        '''
        # reset the database values
        # This is mandatory to avoid wrong cache use through restricted functions
        nlp.database.clear(reset_iteration_counter=True)
        
        # update frozen values with integer values for objective and constraints
        nlp.nonproc_objective.set_frozen_value(integer_values)
        for f in nlp.nonproc_constraints:
            f.set_frozen_value(integer_values)

        return nlp
    
    def solve_dual_problem(self, nlp):
        ''' Solves the dual problem
        '''
        msg = "\n\n######## NLP Solver \n\n"
        LOGGER.info(msg)
        
        cont_sol = OptimizersFactory().execute(nlp, self.algo_NLP,
                          **self.algo_options_NLP#normalize_design_space=False,
                          )
        
        msg = "Continuous solution is "
        msg += str(cont_sol)
        LOGGER.info(msg)
        
#         print("SUB PB HIST")
#         print(nlp.database.get_complete_history(all_iterations=True))
#         print("OVERALL PB HIST")
#         print(self.full_problem.database.get_complete_history(all_iterations=True))
        # add continuous solution to history
        self.cont_solutions.append(cont_sol.f_opt)
        
        return cont_sol
    
    # main Outer Approximation algorithm
    def _termination_criteria(self, ite_nb, mip):
        ''' termination criteria computation
        '''
        if ite_nb == 0:
            _continue = True
        else:
            ub = self.upper_bounds[-1]
            lb = self.lower_bounds[-1]

            if lb >= ub - self.epsilon :
                _continue = False
                msg = "*** Tolerance reached : upper bound vs lower bound ***\n"
                msg += "*** \t Upper Bound (UB) = " + str(self.upper_bounds[-1]) + "\n"
                msg += "*** \t Lower Bound (LB) = " + str(self.lower_bounds[-1]) + "\n"
                msg += "*** \t UB - LB = " + str(ub-lb) #+ " <= " + str(self.epsilon)
                LOGGER.info(msg)
            else:
                _continue = True
        
        return _continue
    
    def update_upper_bounds_history(self, pb):
        """ update the history
        """
        # the current upper bound is the optimal value of the objective of the current NLP
        current_ub = pb.f_opt
        
        # append the objective solution to the upper bounds candidates
        self.upper_bounds_candidates.append(current_ub)
        
        # get the best upper bound found so far (current fopt)
        uk = min(self.upper_bounds_candidates)
        
        # update the upper bound list (fopt history) with the current best upper bound
        self.upper_bounds.append(uk)

## use the main optpb db does not seem to be a good solution since overall iterations are different from NLP ones
#         # store the history to the original problem database
#         store = self.full_problem.database.store
#         val_dicts = {self.UPPER_BOUND_CANDIDATES: current_ub,
#                      self.UPPER_BOUNDS: uk}
#         store(val_dicts, iter=False)
        
    def solve(self):
        ''' Solve the optimization problem : iterative process
        '''
        self.iter_nb = 0
        
        # init integer solution
        xopt_int = self.x0_integer
        
        # initialize NLP(x0_integer)
        nlp = self.build_dual_problem(xopt_int)
        
        # initialize primal problem
        mip = self.build_primal_problem()
        
        while self._termination_criteria(self.iter_nb, mip):
            msg = "\n\n" + "*"*20
            msg += "\nOuterApproximation Iteration %i\n"%self.iter_nb
            msg += "*"*20 + "\n\n"
            LOGGER.info(msg)

            # update NLP(integer solution iteration k)
            nlp = self.update_dual_problem(nlp, xopt_int)

            # compute argmin NLP(integer solution iteration k)
            nlp_sol = self.solve_dual_problem(nlp)
            xopt_cont = nlp_sol.x_opt
            
            # update the full solution vector x
            xsol = self._build_full_vect(xopt_cont, xopt_int)
            
            # update x history
            self.x_solution_history.append(xsol)
            self.update_upper_bounds_history(nlp_sol)
            
            # update primal problem
            uk = self.upper_bounds[self.iter_nb]
            mip = self.update_primal_problem(mip, nlp, uk, xsol)
            
            # solve primal problem
            xopt_int = self.solve_primal_problem(mip)

            LOGGER.info("UPPER BOUNDS")
            LOGGER.info(self.upper_bounds)
            LOGGER.info("LOWER BOUNDS")
            LOGGER.info(self.lower_bounds)
            
            self.iter_nb +=1
        
#         nlp = self.build_dual_pb(xopt_int)
#         nlp_sol = self.solve_dual(nlp)
#         xopt_cont = nlp_sol.x_opt
        
#         xsol = self._build_full_vect(xopt_cont, xopt_int)
#         
#         # update x history
#         self.x_solution_history.append(xsol)
#         self.update_upper_bounds_history(nlp_sol)
#         
#         # update primal problem
#         uk = self.upper_bounds[self.iter_nb]
#         mip = self.update_primal_pb(mip, nlp, uk, xsol)
    
#         self.store_main_history_data()
        
    
#     def build_wrapped_restriction_function(self, mdo_f):
#         """ build a wrapping of the restriction function that stores each call
#         to the database of the main problem
#         """
#         
#         def mdo_f_main_database(xvect):
#             outval = mdo_f(xvect)
#             xv = concatenate((self.xopt_int, xvect))
#             if self.iter_nb == 0:
#                 ubc, ub, lb = 0, 0, 0
#             else:
#                 ubc = self.upper_bounds_candidates[self.iter_nb-1]
#                 ub = self.upper_bounds[self.iter_nb-1]
#                 lb = self.lower_bounds[self.iter_nb-1]
#             name = mdo_f.name.split("_restricted")[0]
#             val_dict = {name: outval,
#                         self.UPPER_BOUND_CANDIDATES: ubc,
#                         self.UPPER_BOUNDS : ub,
#                         self.LOWER_BOUNDS : lb,
#                         self.OA_ITER_NB : self.iter_nb}
#             self.full_problem.database.store(xv, val_dict, add_iter=True)
#             return mdo_f(xvect)
#             
#         return MDOFunction(mdo_f_main_database,
#                             mdo_f.name,
#                             jac=mdo_f._jac,
#                             f_type=mdo_f.f_type,
#                             expr=mdo_f.expr,
#                             args=mdo_f.args,
#                             dim=mdo_f.dim,
#                             outvars=mdo_f.outvars,)
    
    
#     def store_main_history_data(self):
#         """ creates the dataframe where the overal optimization monitoring data is stored
#         """
#         data = {self.UPPER_BOUND_CANDIDATES: self.upper_bounds_candidates,
#                 self.UPPER_BOUNDS: self.upper_bounds,
#                 self.LOWER_BOUNDS: self.lower_bounds}
#         self.opt_history = DataFrame(data)
        
#         LOGGER.info("Integer solution is " + str(self.int_solutions))
#         LOGGER.info("Continuous solution is"  + str(self.cont_solutions))
        
        
                