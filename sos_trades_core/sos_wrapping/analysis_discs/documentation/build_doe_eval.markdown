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
            'process_repository':       folder root of the sub processes to be nested inside the DoE.
                                        If 'None' then it uses the sos_processes python for doe creation.
            'process_name':             selected process folder name to be nested inside the DoE.
                                        If 'None' then it uses the sos_processes python for doe creation.
            'usecase_name' :            either empty or an available usecase of the sub_process
            'usecase_data' :            anonymized dictionary of usecase inputs to be nested in context
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

sos_trades_core.sos_processes.test

	test_disc_hessian
	usecase
	usecase2
	usecase3

	test_proc_build_disc0
	usecase1
	usecase2

	test_proc_build_disc1_all_types
	usecase1
	usecase2
	
	test_proc_build_disc1_grid
	usecase1
	usecase2

	test_disc1_disc2_coupling
	usecase_coupling_2_disc_test
	
	test_sellar_coupling
	usecase
	
	test_disc10_setup_sos_discipline
	usecase_affine
	usecase_linear
	usecase_polynomial

## The several usage steps are detailed below.

###  Step 1 : xxxxx

xxxx

