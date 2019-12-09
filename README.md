[![CircleCI](https://circleci.com/gh/aditya140/ques_gen/tree/master.svg?style=svg)](https://circleci.com/gh/aditya140/ques_gen/tree/master)

# ques_gen
NLP Project 1

# How to run
## Install requirements:
1. ```pip install req.txt ``` (if using blank python env)
   ```pip install requirements.txt``` (if using base env as conda)
2. Adjust the hyper parameters as required in ```hyparameteres.txt``` file.
3. Run ```python train.py```
4. [optional] launch tensorboard in directory ./runs to monitor training
5. run ``` python eval_metrics.py``` to evalutate metrics (saved as ```metrics.txt```)
6. The output of the test dataset will be saved as ```eval_results.csv```
