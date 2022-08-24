# Documentation for DoE_eval driver (process builder)


## Short description
The build doe_eval discipline driver is a specialisation of the sos-eval driver. It extend the doe_eval discipline driver bu providing the capability to select in the GUI the used sub_process of the driver. 

The doe_eval capability allows provide or create a sample and to run the subprocess on this sample. 


## Strucrure of Desc_in/Desc_out
        |_ DESC_IN
            |_ SUB_PROCESS_INPUTS (structuring)
                |_ EVAL_INPUTS (structuring,dynamic : self.sub_proc_build_status != 'Empty_SP') NB: Mandatory not to be empty (If not then warning)
                |_ EVAL_OUTPUTS (structuring,dynamic : self.sub_proc_build_status != 'Empty_SP') NB: Mandatory not to be empty (If not then warning)
                |_ SAMPLING_ALGO (structuring,dynamic : self.sub_proc_build_status != 'Empty_SP')
                        |_ CUSTOM_SAMPLES_DF (dynamic: SAMPLING_ALGO=="CustomDOE") NB: default DESIGN_SPACE depends on EVAL_INPUTS (As to be "Not empty") And Algo 
                        |_ DESIGN_SPACE (dynamic: SAMPLING_ALGO!="CustomDOE") NB: default DESIGN_SPACE depends on EVAL_INPUTS (As to be "Not empty") And Algo
                        |_ ALGO_OPTIONS (structuring, dynamic: SAMPLING_ALGO != None)
            |_ N_PROCESSES
            |_ WAIT_TIME_BETWEEN_FORK
            |_ NS_IN_DF (dynamic: if sub_process_ns_in_build is not None)
        |_ DESC_OUT
            |_ SAMPLES_INPUTS_DF
            |_ <var>_dict (dynamic: sampling_algo!='None' and eval_inputs not empty and eval_outputs not empty, for <var> in eval_outputs)

##     Description of DESC parameters
        SUB_PROCESS_INPUTS: 	    All inputs for driver builder in the form of a dictionary of four keys        
            'PROCESS_REPOSITORY':       folder root of the sub processes to be nested inside the DoE.
                                        If 'None' then it uses the sos_processes python for doe creation.
            'PROCESS_NAME':             selected process folder name to be nested inside the DoE.
                                        If 'None' then it uses the sos_processes python for doe creation.
            'PROCESS_NAME' :            either empty or an available usecase of the sub_process
            'USECASE_DATA' :            anonymized dictionary of usecase inputs to be nested in context
                                        it is a temporary input: it will be put to None as soon as                                                                        
                                        its content is 'loaded' in the dm. We will have it has editable    
        EVAL_INPUTS:                    selection of input variables to be used for the DoE
        EVAL_OUTPUTS:                   selection of output variables to be used for the DoE (the selected observables)
        SAMPLING_ALGO:                  method of defining the sampling input dataset for the variable chosen in self.EVAL_INPUTS
        N_PROCESSES:
        WAIT_TIME_BETWEEN_FORK:
        SAMPLES_INPUTS_DF :             copy of the generated or provided input sample
        ALL_NS_DICT :                   a map of ns keys: values
        CUSTOM_SAMPLES_DF:              provided input sample
        DESIGN_SPACE:                   provided design space
        ALGO_OPTIONS:                   options depending of the choice of self.SAMPLING_ALGO
        <var observable name>_dict':    for each selected output observable doe result
                                        associated to sample and the selected observable

## The several usage steps are detailed below.
Example of models

    ########################################################
    sos_trades_core.sos_processes.test
    test_disc_hessian

    usecase1
    {'<study_ph>.Hessian.x': 2.0,
    '<study_ph>.Hessian.y': 3.0,
    '<study_ph>.Hessian.ax2': 4.0,
    '<study_ph>.Hessian.by2': 5.0,
    '<study_ph>.Hessian.cx': 6.0,
    '<study_ph>.Hessian.dy': 7.0,
    '<study_ph>.Hessian.exy': 12.0}
    
    usecase2
    {'<study_ph>.Hessian.x': 12.0,
    '<study_ph>.Hessian.y': 13.0,
    '<study_ph>.Hessian.ax2': 14.0,
    '<study_ph>.Hessian.by2': 15.0,
    '<study_ph>.Hessian.cx': 16.0,
    '<study_ph>.Hessian.dy': 17.0,
    '<study_ph>.Hessian.exy': 112.0}

    usecase3
    {'<study_ph>.Hessian.x': 22.0,
    '<study_ph>.Hessian.y': 23.0,
    '<study_ph>.Hessian.ax2': 24.0,
    '<study_ph>.Hessian.by2': 25.0,
    '<study_ph>.Hessian.cx': 26.0,
    '<study_ph>.Hessian.dy': 27.0,
    '<study_ph>.Hessian.exy': 212.0}

    ###################################################
    sos_trades_core.sos_processes.test
    test_proc_build_disc0

    usecase_int
    {'<study_ph>.Disc0.r': 4,
    '<study_ph>.Disc0.mod': 2}

    usecase_float
    {'<study_ph>.Disc0.r': 4.5,
    '<study_ph>.Disc0.mod': 2}

    #####################################################
    sos_trades_core.sos_processes.test
    test_proc_build_disc1_all_types

    usecase1
    {'<study_ph>.Disc1.x': 5.5,
    '<study_ph>.Disc1.a': 3,
    '<study_ph>.Disc1.b': 2,
    '<study_ph>.Disc1.name': 'A1',
    '<study_ph>.Disc1.x_dict': {}}

    usecase2
    {'<study_ph>.Disc1.x': 5.5,
    '<study_ph>.Disc1.a': 3,
    '<study_ph>.Disc1.b': 2,
    '<study_ph>.Disc1.name': 'A1',
    '<study_ph>.Disc1.x_dict': {'x_1':1.1,'x_2':2.1,'x_3':5.5,'x_4':9.1}}

    #####################################################
    sos_trades_core.sos_processes.test
    test_proc_build_disc1_grid

    usecase1
    {'<study_ph>.Disc1.x': 3.5,
    '<study_ph>.Disc1.a': 20,
    '<study_ph>.Disc1.b': 2,
    '<study_ph>.Disc1.name': 'A1',
    '<study_ph>.Disc1.x_dict': {'x_1': 1.1, 'x_2': 2.1, 'x_3': 3.5, 'x_4': 9.1},
    '<study_ph>.Disc1.d': 1.,
    '<study_ph>.Disc1.f': 1.,
    '<study_ph>.Disc1.g': 1.,
    '<study_ph>.Disc1.h': 1.,
    '<study_ph>.Disc1.j': 1.}

    usecase2
    {'<study_ph>.Disc1.x': 13.,
    '<study_ph>.Disc1.a': 30,
    '<study_ph>.Disc1.b': 5,
    '<study_ph>.Disc1.name': 'A1',
    '<study_ph>.Disc1.x_dict': {'x_1': 1.2, 'x_2': 2.2, 'x_3': 13., 'x_4': 9.1},
    '<study_ph>.Disc1.d': 1.,
    '<study_ph>.Disc1.f': 1.,
    '<study_ph>.Disc1.g': 1.,
    '<study_ph>.Disc1.h': 1.,
    '<study_ph>.Disc1.j': 1.}

    ##################################################### 
    sos_trades_core.sos_processes.test
    test_disc1_disc2_coupling

    usecase_coupling_2_disc_test
    {'<study_ph>.x': 10.,
    '<study_ph>.Disc1.a': 5.,
    '<study_ph>.Disc1.b': 25431.,
    '<study_ph>.y': 4.,
    '<study_ph>.Disc2.constant': 3.1416,
    '<study_ph>.Disc2.power': 2}

    ##################################################### 
    sos_trades_core.sos_processes.test
    test_sellar_coupling

    usecase
    {'<study_ph>.SellarCoupling.x': array([1.]),
    '<study_ph>.SellarCoupling.y_1': array([1.]),
    '<study_ph>.SellarCoupling.y_2': array([1.]),
    '<study_ph>.SellarCoupling.z': array([1., 1.]),
    '<study_ph>.SellarCoupling.Sellar_Problem.local_dv': 10.,
    '<study_ph>.SellarCoupling.max_mda_iter': 100,
    '<study_ph>.SellarCoupling.tolerance': 1e-12}

    #####################################################
    sos_trades_core.sos_processes.test
    test_disc10_setup_sos_discipline

    usecase_affine
    {'<study_ph>.Disc10.Model_Type': 'Affine',
    '<study_ph>.Disc10.x': 5.0,
    '<study_ph>.Disc10.b': 3.0}

    usecase_linear
    {'<study_ph>.Disc10.Model_Type': 'Linear',
    '<study_ph>.Disc10.x': 2.0}

    usecase_polynomial
    {'<study_ph>.Disc10.Model_Type': 'Polynomial',
    '<study_ph>.Disc10.x': 2.0,
    '<study_ph>.Disc10.b': 4.0}


## The several usage steps are detailed below.

###  Step 1 : xxxxx

xxxx

