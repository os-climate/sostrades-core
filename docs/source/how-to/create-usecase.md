# Create a usecase

To create a use case and provide input data for your process in the SoSTrades platform, follow these steps:

Create a **usecase.py** in a process directory.

Each usecase contains a set of input data to run on a given process.

The file usecase.py implements a class **Study** based on **StudyManager**.

This class implements a method **setup_usecase()** that returns a dictionary (or list of dictionaries), with all inputs needed for execution of the usecase.

In the main, the study is instantiated, asked to load data and to run.


```python
from sostrades_core.study_manager.study_manager import StudyManager
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

    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()

```
## Test your usecase

All use cases can be accessed via the reference management page on the SoSTrades platform. Subsequently, each use
case within a repository undergoes thorough testing to ensure proper configuration, is executed twice, and consistently
reproduces the same results. Additionally, all post-processing linked to the use case (the one in all the wrapp of the
associated process) are validated .
Verify the robustness of your use case with the following command:

```python


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()

```

## Visualize the post-processings of your usecase

A dedicated factory (PostprocessingFactory) manages post-processing functionalities within the SoSTrades platform. Users have the option to preview all post-processings created for their usecase (in all the wrapp of the associated process but also at a given node, see how to create-postprocessing for mroe details) with these commands : 

```python


if '__main__' == __name__:
    from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
    ppf = PostProcessingFactory()
    graph_list = ppf.get_all_post_processings(uc_cls.ee, False)
    for graph in graph_list:
       graph.to_plotly().show()

```
The "show" method will open a window in your preferred web browser for each post-processing graph associated with the use case. 
There may be variations in typographies or policies compared to the SoSTrades GUI rendering.