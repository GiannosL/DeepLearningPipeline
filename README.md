<h1>Multi-layer perceptron for classification pipeline</h1>

<h2>Introduction</h2>
<p>
    This is a pipeline which automates classification projects using a simple multi-layered perceptron. 
    The pipeline has some basic data exploration features along with the option to generate some plots. 
    The prediction results come along with some rudimentary statistics as well as the ability to vizualized them easily.
    This pipeline was built with the goal of providing a template which can be easily expanded upon.
</p>

<p>
    The deep-learning infrastructure used in the pipeline is based on the <em>PyTorch</em> framework. It 
    consists of a Deep Neural Network with three hidden layers. The number of input nodes is automatically
    adjusted to the number of input features and the number of output nodes is automatically adjusted by the number of classes.
    <em>At the moment the model can only use continuous variables as input!</em>
</p>

<p>
    This script runs on <strong>Python3</strong>.
    In order to use this tool you need to first install the following packages in your python environment:
    <ul>
        <li>NumPy</li>
        <li>Pandas</li>
        <li>MatPlotLib</li>
        <li>PyTorch</li>
        <li>SciKit-learn</li>
    </ul>
</p>

<h2>Use guide</h2>

<h3>Data set-up</h3>

<p>
    In order to use the pipeline, the input data needs to be organised in a specific (yet simple) way.
    In my repository I am using a heart-disease dataset as an example to demonstrate how the input files 
    should look. The dataset can be downloaded from Kaggle: 
</p>

[Heart Disease Dataset](https://www.kaggle.com/datasets/yasserh/heart-disease-dataset?resource=download)

<p>
    I am also providing the iris-dataset which was initially used to develop the model but it only contains 
    continuous variables.
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

    target: "/path/to/target_file.csv"

testing:
    <feature_1>: "/path/to/file_1.csv"
    <feature_2>: "/path/to/file_2.csv"

    target: "/path/to/target_file.csv"
````

<p>
    The target file should contain the class for each individual encoded as 0,1,... <br>
    The variable pointing to the target file should be named <strong>"target"</strong> always. 
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
The pipeline has a few command line arguments (currently only one) to help with automation when running. The 
argument <em>--configfile \<path_to_file\></em> receives the path to the yaml configuration file.


```
python3 main.py --configfile <path/to/file>
```


<h2>Model details</h2>
<p>
    The neural network has consists of an input layer, three hidden layers and an output layer. The number of nodes 
    in the input layer is equal to the number of features used as input and the number of nodes in the output layer is 
    equal to the final number of classes. The number of nodes per hidden layer is calculated during the process of 
    <em>hyper-parameter optimization</em>.
</p>

<p>
    The step of hyper-parameter optimization (HPO) preceeds the model training. HPO is implemented with the use of
    the Optuna framework. At the moment only the number of nodes per hidden layer is treated as a hyper-parameter. 
</p>