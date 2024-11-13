# Biologically Inspired Computation
## Particle Swarm Optimisation
### Ben Johnston, Walid Dahnn

project introduction -- todo

# How to use

To run the particle swarm optimiser algorithm simple run
`Experiment.run_experiment_avg()` from the [Experiment](Implementation/Experiment.py) class,
the [main.py](Implementation/main.py) file can be run for quick and easy run with all values setup allready

### parameters include 

hyperparameters file path `hyper_path` (json formatting)

training and testing dataset `train_path` (csv formatting)

saving the outputs `save` True or False

overide [hyperparameters.json](Data/hyperparameters.json) `overide_hp` (json formatting)

# Experiment Outputs

all experiments will be outputed to [Experiments](Experiments) an example of an output would look like this ...
![example output](Experiments/25_500_sigmoid_12_0,9_0,8_0,5_0,05.png)

# GUI
there is also a GUI that can be run in the browser for easy single run testing and easy json file editing, just run the
[GUI.py](GUI/GUI.py) script and the browser should load 
![gui example](GuiExampleImage.png)

