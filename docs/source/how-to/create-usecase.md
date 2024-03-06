# Create a usecase

How to create a use case to feed input data to your process and run it.

Create a **usecase.py** in a process directory.

Each usecase contains a set of input data to run on a given process.

The file usecase.py implements a class **Study** based on **StudyManager**.

This class implements a method **setup_usecase()** that returns a dictionary (or list of dictionaries), with all inputs needed for execution of the usecase.

In the main, the study is instantiated, asked to load data and to run.


```python

class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self, study_folder_path=None):
        dict_values = {
            'usecase.x': 3,
            'usecase.Disc1.a': 1,
            'usecase.Disc1.b': 5,
        }
        return dict_values

if '__main__' == __name__:
    start = time.time()
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
    stop = time.time()
    print(stop-start)

```
