from gemseo.core.discipline import MDODiscipline


class DisciplineProxy(object):
    '''
    classdocs
    '''


    def __init__(self, mode):
        '''
        Constructor
        mode (type SoSDiscipline / MDODiscipline)
        '''
        self.builder = None
        self.mdo_discipline = None
        self.mode = 
        
    # method names common with GEMSEO
        
    def get_input_data_names(self):
        ''' returns the list of input data names,
        based on i/o and namespaces declarations in the user wrapper
        '''
        
    def get_output_data_names(self):
        ''' returns the list of input data names,
        based on i/o and namespaces declarations in the user wrapper
        '''
    
    # sostrades methods, linked to configuration and execution preparation steps
    
    def prepare_execution(self):
        '''
        Instantiate a MDODiscipline
        '''
        disc = MDODiscipline(name, grammar_type=self.SOS_GRAMMAR_TYPE)
#         self.reload_io()
#         
        self.update_gems_grammar_with_data_io(disc)
        
        setattr(disc, "_run", *._run)
        self.mdo_discipline = disc
        
    def reload_io(self):
        '''
        Create the data_in and data_out of the discipline with the DESC_IN/DESC_OUT, inst_desc_in/inst_desc_out
        and initialize GEMS grammar with it (with a filter for specific variable types)
        '''
        # set input/output data descriptions if data_in and data_out are empty
        self.create_data_io_from_desc_io()

        # update data_in/data_out with inst_desc_in/inst_desc_out
        self.update_data_io_with_inst_desc_io()

    def update_dm_with_data_dict(self, data_dict):
        self.dm.update_with_discipline_dict(
            self.disc_id, data_dict)

    def update_gems_grammar_with_data_io(self):
        # Remove unavailable GEMS type variables before initialize
        # input_grammar
        if not self.is_sos_coupling:
            data_in = self.get_data_io_with_full_name(
                self.IO_TYPE_IN)
            data_out = self.get_data_io_with_full_name(
                self.IO_TYPE_OUT)
            self.init_gems_grammar(data_in, self.IO_TYPE_IN)
            self.init_gems_grammar(data_out, self.IO_TYPE_OUT)

    def get_sosdisc_inputs(self, keys=None, in_dict=False, full_name=False):
        """Accessor for the inputs values as a list or dict

        :param keys: the input short names list
        :param in_dict: if output format is dict
        :param full_name: if keys in output are full names
        :returns: the inputs values list or dict
        """

        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            keys = list(self.get_data_in().keys())
            in_dict = True
        inputs = self._get_sosdisc_io(
            keys, io_type=self.IO_TYPE_IN, full_name=full_name)
        if in_dict:
            # return inputs in an dictionary
            return inputs
        else:
            # return inputs in an ordered tuple (default)
            if len(inputs) > 1:
                return list(inputs.values())
            else:
                return list(inputs.values())[0]

    def get_sosdisc_outputs(self, keys=None, in_dict=False, full_name=False):
        """Accessor for the outputs values as a list or dict

        :param keys: the output short names list
        :param in_dict: if output format is dict
        :param full_name: if keys in output are full names
        :returns: the outputs values list or dict
        """
        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            keys = [d[self.VAR_NAME] for d in self.get_data_out().values()]
            in_dict = True
        outputs = self._get_sosdisc_io(
            keys, io_type=self.IO_TYPE_OUT, full_name=full_name)
        if in_dict:
            # return outputs in an dictionary
            return outputs
        else:
            # return outputs in an ordered tuple (default)
            if len(outputs) > 1:
                return list(outputs.values())
            else:
                return list(outputs.values())[0]
        
        
    # Overloaded by subclasses (user)
    
    def setup_disciplines(self):
        pass
    
    def configure(self):
        '''
        No MDODiscipline instantiation here
        Prepare all information necessary to instantiate MDODiscipline
        '''
        pass

    def build(self):
        '''
        '''
        pass
        
    def run(self):
        '''
        Implemented by the user
        '''
        pass
