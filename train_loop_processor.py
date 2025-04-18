import subprocess
import itertools

#Example Manual Model Train Running
# python model_auto_trainer_v1.py --learning_rate 5e-5 --epochs 20 --batch_size 16 --model bert-base-uncased 

#Models Available:
MODEL_NAME = 'bert-base-uncased' #(bert)
# MODEL_NAME = 'bert-base-multilingual-cased' #(mBERT)
# MODEL_NAME = 'xlm-roberta-base' #(XLM-RoBERTa, Distil Roberta) 1e-5 for xlm notried yet
#MODEL_NAME = "google-bert/bert-base-cased" #(mobileBert)

params_list = [
    {"learning_rate": 2e-5, "epochs": 10, "batch_size": 16},
    {"learning_rate": 5e-5, "epochs": 8, "batch_size": 64},
    {"learning_rate": 3e-5, "epochs": 15, "batch_size": 8},
]


hp_grid = {
    "learning_rate": [2e-5, 5e-5,3e-5],
    "epochs": [100],
    "batch_size": [16, 32, 64, 128]
}

# To Generate all possible hyperparameter combinations
keys, values = zip(*hp_grid.items())
all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
params_list = all_combinations;

for params in params_list:
    command = f"python model_auto_trainer_v1.py --learning_rate {params['learning_rate']} --epochs {params['epochs']} --batch_size {params['batch_size']} --model {MODEL_NAME}"
    print("|-------------------------------------------------------------------|")
    subprocess.run(command, shell=True)


MODEL_NAME = 'xlm-roberta-base' #(XLM-RoBERTa, Distil Roberta)
for params in params_list:
    command = f"python model_auto_trainer_v1.py --learning_rate {params['learning_rate']} --epochs {params['epochs']} --batch_size {params['batch_size']} --model {MODEL_NAME}"
    print("|-------------------------------------------------------------------|")
    subprocess.run(command, shell=True)