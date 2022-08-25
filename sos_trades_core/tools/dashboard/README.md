# Dashboard

The dashboard is a process summary to display main lines / inputs / results of a study.
There is a unique dashboard template for a process, that contains all elements to display. 
Then, the dashboard of a study is generating, following the dashboard template of the study's process, with the study data.


## Dashboard template

The dashboard template is a json file, containing information about how is disposed the view of the process and what texts, parameters or post-processing are displayed.
This file named **dashboard_template.json** must be located in a **dashboard** folder in the process repository (similar to documentation folder).

### Dashboard template format

The content of the dashboard template is in the json format. 
It should contain:
 - a title
 - a list of rows

A row contains a list of elements that will be displayed on the same line of the dashboard. So beware of the number of element you set on a row, it may overflow of the screen.
A row may contain:
- a parameter, 
- a chart,
- a text,
- a section separator.

A PARAMETER element is composed by the following attributes:
- content_type : "SCALAR",
- content_namespace : the content_namespace contains the namespace of the parameter to retrieve the value to be displayed.
The namespace has to be formatted so that the name of the study is replaced by "{study_name}" (see example below)
The parameters with dataframe values


A CHART element is composed by the following attributes:
- content_type : "POST_PROCESSING"
- content_namespace : the content_namespace contains the namespace of the discipline where the chart is.
The namespace has to be formatted so that the name of the study is replaced by "{study_name}" (see example below)
- graph_index : index of the graph in the discipline, starting at 0 (same order than in the tab "post-processing" of the study in the gui)

A TEXT element is composed by the following attributes:
- content_type : "TEXT",
- content : the text to display, it can be written in html to add display ornaments (see example below).
- style (optional): The css class style of the text element (see example below).

A LINE separator element is composed by the following attributes:
- content_type : "TEXT",
- style : "hr-solid"

### Dashboard generation

The dashboard is generated after each saving of the study and only if the root process status is at DONE:
- after the creation of a study
- after an execution of the study

If the dashboard is available, it will be displayed in the tab "Dashboard".

### Dashboard template example

Here is an example of a dashboard template with comments and explanations surrounded by "//". The content is written directly in this section, but an example of the complete template file is available next to this README.md file in the dashboard folder.

    {
      "title": "example of dashboard", //The title of the dashboard//
      "rows": [ 

        // the first row//
        [ 
          //This is the first TEXT element //
          { 
            "content_type": "TEXT",
            "content": "This is a process of generic value assessment ",
          }
        //end of the first row//
        ],

        //second row//
        [
          // a line separator element//
          {
            "content_type": "TEXT",
            "style": "hr-solid"
          }
        ],
    
        //3rd row//
        [
          // a sub title element//
          {
            "content_type": "TEXT",
            "content": "Inputs",
            "style": "h3" //it is the css class name of a title of 3rd level //
          }
        ],
    
        //4th row//
        [
          // a parameter element //
          {
            "content_type": "SCALAR",
            "content_namespace": "{study_name}.Ratatouille.launch_year"
          }
          ,
          // a 2nd parameter element on the same line//
          {
            "content_type": "SCALAR",
            "content_namespace": "{study_name}.Tomato_sauce.launch_year"
          }
        ],
    
        // a line separator element in a new row//
        [
          {
            "content_type": "TEXT",
            "style": "hr-solid"
          }
        ],
        [
          // a chart//
          {
            "content_type": "POST_PROCESSING",
            "content_namespace": "{study_name}.OpEx", // the chart is in the OpEx discipline //
            "graph_index": 0 // This is the first chart of the discipline //
          }
        ]
      ]
    }

### Dashboard rendering

The text element will be displayed with the css class style defined (like title if the style "h1" is applied).

The parameter element will be displayed with the parameter id in bold following with the parameter value.

The post-processing element will be display with a plotly chart as it is displayed on the GUI but with only the default Sostrades logo and without the pareto chart options.

### Dashboard error management

If a parameter is not found, the massage "{name of the parameter} is not found" is displayed.
If a mandatory element is not in the template (for example if there is no "namespace_content" in a "PARAMETER" element section), the dashboard will be in error and not be displayed.