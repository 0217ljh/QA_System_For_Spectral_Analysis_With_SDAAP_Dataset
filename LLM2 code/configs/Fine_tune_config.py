import torch.cuda
import torch.backends
import os
from typing import List, Dict


Configs = { 'training hyperparams':{
                                'batch_size' : 128, 
                                'micro_batch_size' : 4,
                                'num_epochs' : 3,
                                'learning_rate' : 3e-4,
                                'cutoff_len' : 356,
                                'val_set_size' : 200,  
                                'resume_from_checkpoint' : None,  # either training checkpoint or final adapter
                                'prompt_template_name' : "alpaca",
                                'warmup_steps' : 100,
                                'eval_steps' : 20,
                                'save_steps' : 20,
                                'save_total_limit': 5
                                },

           'lora hyperparams':{
                                'lora_r' : 16,
                                'lora_alpha' : 32,
                                'lora_dropout' : 0.05,
                                'lora_target_modules' : [
                                    "q_proj",
                                    "v_proj",
                                    "k_proj",],
                                },

            'llm hyperparams':{
                                'train_on_inputs' : True, # if False, masks out inputs in loss
                                'add_eos_token' : False,
                                'group_by_length' : False, # faster, but produces an odd training loss curve
                                },

           'wandb params':{
                                'wandb_project' : "test",
                                'wandb_run_name' : "",
                                'wandb_watch' : "",  # options: false | gradients | all
                                'wandb_log_model' : "",  # options: false | true
                                }
           }  # The prompt template to use, will default to alpaca.


