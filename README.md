## CAT pipeline 

after adjusting the arguments in lora_train.py, run 

```bash
accelerate launch vanila_lora_train.py
```

for inference 

```bash
python lora_test.py
```

# specific training and checkpoint saving

first set a directory to save your config, validation img and lora weight

you can use results dir already in the reposit

then for example 

```bash
mkdir ./results/pokemon_vanila_02012024
#create a config file for this training 
#set the output dir to the dir above
#set max_train_step and checkpointing_steps
#then for each checkpointing_steps, the program will save lora
#also validation prompts can be added
touch ./results/pokemon_vanila_02012024/tuning_config.json
#then run train 
accelerate launch vanila_lora_train.py --tuning_config_path ./results/pokemon_vanila_02012024/tuning_config.json
#also add 
CUDA_VISIBLE_DEVICES=2 
#in front of the command above to control cuda devices
#then change the folder dir of load_lora_weights to checkpoint-number folder in vanila_lora_test to continue inference
```