<h1>Multi-perceptron for classification pipeline</h1>

<h2>Introduction</h2>
<p>
This is a pipeline which automates classification projects using a simple multi-layered perceptron. 
The pipeline has some basic data exploration features along with the option to generate some plots. 
The prediction results come along with some rudimentary statistics as well as the ability to vizualized them easily.
 This pipeline was built with the goal of providing a template which can be easily expanded upon.
</p>
<p>
The deep-learning infrastructure used in the pipeline is based on the <em>PyTorch</em> framework.
</p>

<br> <hr>
<h2>Use guide</h2>

<h3>Data set-up</h3>

<p>
In order to use the pipeline, the input data needs to be organised in a specific (yet simple) way.
In my repository I am using the iris dataset as an exampleto demonstrate how the input files
 should look.
</p>

<p>
Simply put, after performing the train-test split of your dataset you should create a 
<em>.csv file</em> for each of the features. The features for the training dataset should
go into directory under datasets called "training" (<em>"dataset/training"</em>) and the test
data should go under <em>"dataset/testing"</em>.
</p>

<h4>features.yaml</h4>
<p>
Under the "dataset" directory a YAML file should contain the paths to the input features in the
following format:
</p>

````
training:
    <feature_1>: "/path/to/file_1.csv"
    <feature_2>: "/path/to/file_2.csv"

    condition: "/path/to/target_file.csv"

testing:
    <feature_1>: "/path/to/file_1.csv"
    <feature_2>: "/path/to/file_2.csv"

    condition: "/path/to/target_file.csv"
````

<p>
The target file should contain the class for each individual encoded as 0,1,... <br>
The variable pointing to the target file should be named <strong>"condition"</strong> always. 
The variables for the features do not matter in our naming scheme.
</p>


<h3>Configuration</h3>

<p>
The simplest use-case for the pipeline is to run it through the configuration file leaving everything in 
the <em>main.py</em> file default. The configuration file, named <strong>model_configuration.yaml</strong>
contains <em>two</em> variables:
</p>

</ul>
    <li><strong>work_directory [str]:</strong> (absolute) path to the pipeline's output directory.</li>
    <li><strong>features [str]:</strong> a comma separated list of the variables (their names should be 
        in exactly the same as in the "dataset/features.yaml" file).</li>
</ul>

<h3>Command</h3>

```
python3 main.py
```
