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
