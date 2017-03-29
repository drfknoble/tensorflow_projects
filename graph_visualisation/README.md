# graph_visualisation

In this project, I show you how to visualise a TensorFlow graph. We'll start simple and show how we can make a basic graph, compute some basic equations, then display what our graph looks like via TensorBoard.

# Table of Contents

1. Requirements.
1. Installation.
1. Usage.
1. Contributions.
1. Credits.

# Requirements


I have used the following tools and libraries to create and run each project:

1. [TensorFlow](https://www.tensorflow.org/).
1. [Anaconda](https://www.continuum.io/).
1. [Nvidia CUDA](https://developer.nvidia.com/cuda-zone).
1. [Nvidia CUDNN](https://developer.nvidia.com/cudnn).
1. [Microsoft Visual Studio Code (Code)](https://code.visualstudio.com/).
1. [Code Python Extension](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python).

The project has been compiled and run on PCs running Windows 7 x64 and Windows 10 x64 to ensure they work.

Please note: if you want to use TensorFlow on a GPU, you'll need to make sure your GPU supports Nvidia's CUDA, and install CUDA and CUDNN.

# Installation

The following is a list of steps to follow to get started with using this project':

1. Clone the tensorflow_projects repo'.
1. Install Anaconda and create a Python 3.5 environment for TensorFlow, e.g. 'tf' for TensorFlow running on a CPU or 'tf-gpu' for TensorFlow running on a GPU. Activate the environment and install jupyer, matplotlib, scipy, and tensorflow/tensorflow-gpu.
1. Install Code and the Python extension. 
1. Open a Windows command line and activate an environment. Type 'code' to open Code, then browse to the graph_visualisation directory. Open main.py.
1. Use Code's built-in console and type 'python main.py' to run the script.

At this point, everything should compile and run.

Please note: you can also run main.py from within Code by typing 'F1', then 'run'.

# Usage 

To run the main.py script, browse to getting_started directory and run the following command in a command line window:

    python main.py

To run TensorBoard, run the following command:

    tensorboard --logdir=./logs

You should see the following outputs:

![Scalars](./data/output/tb_scalars.png)

![Graph](./data/output/tb_graphs.png)

# Contributions

If you feel like you can make a contribution; please, feel free to make a request.

# Credits

Dr. Frazer K. Noble. 
 
Follow me on Twitter at [@FrazerNoble](https://twitter.com/FrazerNoble).
