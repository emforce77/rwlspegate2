
# SPEGT


## Framework
A detailed description can be found through the link. <https://github.com/emforce77/rwlspegate2/blob/main/technical_report.pdf>



## Results

Dataset       | #layers | #params | Metric         | Valid           | Test           |
--------------|---------|---------|----------------|-----------------|----------------|
PCQM4M-V2     | 24      | 90.1M   | MAE            | 0.0863          | 0.0868         |


## Requirements

* `python == 3.7`
* `pytorch == 1.12.1`
* `numpy == 1.23.5`
* `numba == 0.56.4`
* `ogb == 1.3.2`
* `rdkit==2019.03.1`
* `yaml == 0.2.5`
* 'network == 2.8.4'

## Run Training and Evaluations

You can specify the training/prediction/evaluation configurations by creating a `yaml` config file and also by passing a series of `yaml` readable arguments. (Any additional config passed as argument willl override the config specified in the file.)

* To run training: ```python run_training.py [config_file.yaml] ['config1: value1'] ['config2: value2'] ...```
* To perform evaluations: ```python do_evaluations.py [config_file.yaml] ['config1: value1'] ['config2: value2'] ...```

Config files for the results can be found in the configs directory. Examples:
```
python run_training.py configs/pcqm4mv2/egt_90m.yaml
python do_evaluations.py configs/pcqm4mv2/egt_90m.yaml
```


## Python Environment

The Anaconda environment in which the experiments were conducted is specified in the `environment.yml` file.

## Acknowledgements
Our overall experimental pipeline is based on EGT, MPGNNs-LSPE repository. For baseline construction. We appreciate the authors who Hussain et al. and Dwivedi et al. for sharing their code.


