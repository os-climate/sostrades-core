# Documentation for DoE_eval driver (process builder)


## Short description
The build doe_eval discipline driver is a specialisation of the sos-eval driver. It extend the doe_eval discipline driver bu providing the capability to select in the GUI the used sub_process of the driver. 

The doe_eval capability allows provide or create a sample and to run the subprocess on this sample. 

<!---  <i class="fas fa-screwdriver-wrench fa-fw" style="color: fi (Embedding FontAwesome Icons Test) -->  
<!--- @icn-fa-screwdriver-wrench (Embedding FontAwesome Icons Test)  --> 

## Strucrure of Desc_in/Desc_out
        |_ DESC_IN
            |_ REPO_OF_SUB_PROCESSES (structuring)
                        |_ SUB_PROCESS_NAME (structuring)
                                    |_ USECASE_OF_SUB_PROCESS (structuring,dynamic: SUB_PROCESS_NAME!='None')
                                    |_ EVAL_INPUTS (structuring,dynamic : SUB_PROCESS_NAME!='None') NB: Mandatory not to be empty (If not then warning)
                                    |_ EVAL_OUTPUTS (structuring,dynamic : SUB_PROCESS_NAME!='None') NB: Mandatory not to be empty (If not then warning)
                                    |_ SAMPLING_ALGO (structuring,dynamic : SUB_PROCESS_NAME!='None')
                                            |_ CUSTOM_SAMPLES_DF (dynamic: SAMPLING_ALGO=="CustomDOE") NB: default DESIGN_SPACE depends on EVAL_INPUTS (As to be "Not empty") And Algo 
                                            |_ DESIGN_SPACE (dynamic: SAMPLING_ALGO!="CustomDOE") NB: default DESIGN_SPACE depends on EVAL_INPUTS (As to be "Not empty") And Algo
                                            |_ ALGO_OPTIONS (structuring, dynamic: SAMPLING_ALGO != None)
            |_ N_PROCESSES
            |_ WAIT_TIME_BETWEEN_FORK
        |_ DESC_OUT
            |_ SAMPLES_INPUTS_DF
            |_ ALL_NS_DICT
            |_ <var>_dict (dynamic: sampling_algo!='None' and eval_inputs not empty and eval_outputs not empty, for <var> in eval_outputs)

##     Description of DESC parameters
        REPO_OF_SUB_PROCESSES:          folder root of the sub processes to be nested inside the DoE.
                                        If 'None' then it uses the sos_processes python for doe creation.
        SUB_PROCESS_NAME:               selected process folder name to be nested inside the DoE.
                                        If 'None' then it uses the sos_processes python for doe creation.
        EVAL_INPUTS:                    selection of input variables to be used for the DoE
        EVAL_OUTPUTS:                   selection of output variables to be used for the DoE (the selected observables)
        SAMPLING_ALGO:                  method of defining the sampling input dataset for the variable chosen in self.EVAL_INPUTS
        N_PROCESSES:
        WAIT_TIME_BETWEEN_FORK:
        SAMPLES_INPUTS_DF :             copy of the generated or provided input sample
        ALL_NS_DICT :                   a map of ns keys: values
        USECASE_OF_SUB_PROCESS :        either empty or an available usecase of the sub_process
        CUSTOM_SAMPLES_DF:              provided input sample
        DESIGN_SPACE:                   provided design space
        ALGO_OPTIONS:                   options depending of the choice of self.SAMPLING_ALGO
        <var observable name>_dict':    for each selected output observable doe result
                                        associated to sample and the selected observable




## The several usage steps are detailed below.

###  Step 1 : xxxxx

xxxx

