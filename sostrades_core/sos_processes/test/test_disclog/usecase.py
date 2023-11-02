from sostrades_core.study_manager.study_manager import StudyManager
import time


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):

        dict_values = {
            'usecase.DiscLog.log_lines': 10,
            'usecase.DiscLog.wait_time_s': 5,
            }
        return dict_values


if '__main__' == __name__:
    start = time.time()
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
    stop = time.time()
    print(stop - start)
