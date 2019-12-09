[![CircleCI](https://circleci.com/gh/aditya140/ques_gen/tree/master.svg?style=svg)](https://circleci.com/gh/aditya140/ques_gen/tree/master)

# ques_gen
NLP Project 1

# How to run on colab
open the ```Nlp.ipynb``` file in the repo on Google Colab

# How to run on local
## Install requirements:
1. ```pip install -r req.txt ``` (if using blank python env)
   ```pip install -r requirements.txt``` (if using base env as conda)
2. Run ```bash setup.sh```
3. Adjust the hyper parameters as required in ```hyparameteres.txt``` file.
4. Run ```python train.py```
5. [optional] ```tensorboard --logdir runs``` to monitor training
6. run ``` python eval_metrics.py``` to evalutate metrics (saved as ```metrics.txt```)
7. The output of the test dataset will be saved as ```eval_results.csv```
