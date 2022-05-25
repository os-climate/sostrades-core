# Test DoE of a Hessian Discipline
This "Test DoE of Hessian Discipline" specifies an example of a DoE : a process to instantiate the DoE without any nested builder or by specifiying the nested builder from a process.py python file.
As possible values it is currently restricted to the test_disc_hessian process that instantiates the Hessian Discipline (the wrapped discipline sos_trades_core.sos_wrapping.test_discs.disc_hessian.DiscHessian).
The Hessian discipline as only local input/output variable. There is no use of namespace for the nested builder.
