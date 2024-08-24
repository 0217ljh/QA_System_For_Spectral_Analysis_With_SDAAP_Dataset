import os
import sys
import fire
import torch
import evaluate
from IPython.display import display, HTML
from datasets import load_dataset, Dataset, DatasetDict
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM


import datetime
import wandb
import bitsandbytes
import collections
import pandas as pd
import torch.nn as nn
import numpy as np
from typing import List, Dict,Optional,Union
from utils.prompter import Prompter
from sklearn.model_selection import train_test_split
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
#os.environ["WANDB_API_KEY"] = 'de1a505eeff3e69c555e57324f9dfa10002dd1f8'

class Fine_tune():
    def __init__(self, train_data_path,  test_data=None, **Configs):
        # Initialisation Configuration
        self.train_data = None
        self.test_data = None
        self.val_data = None
        
        self.data_preprocess = None
        self.data_map = None

        self.tokenizer_name = None
        self.tokenizer = None
        self.prompter = None
        self.model = None
        self.default_model = 'meta-llama/Meta-Llama-3-8B'

        self.trainer = None
        self.Arguments = None

        self.default_config = Configs
        self.current_config = self.default_config.copy()

        for key, value in Configs.items():
             setattr(self, str(key), value)

        self.base_model = self.default_model 

        if train_data_path is None:
            raise ValueError("train_data_path can't be None")
        else:self.train_data_path = train_data_path
        self.test_data = test_data

        current_datetime = datetime.datetime.now()
        current_date = current_datetime.strftime("%Y-%m-%d/")
        current_time = current_datetime.strftime("%H-%M")
        self.output_dir = default_path = os.path.join('./Train/Output/', current_date) + current_time
        self.wandb_initiate = None

        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size   # 128 // 4 = 32 ,向下取整
        self.device_map = "auto"
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))   # 这个环境变量一般为1，不知道具体有什么用，可能和分卡训练有关。
        self.ddp = self.world_size != 1   # ddp一般为False
        if self.ddp:
            self.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            self.gradient_accumulation_steps = self.gradient_accumulation_steps // self.world_size

        self.default_config.update({'gradient_accumulation_steps':self.gradient_accumulation_steps,
                                    'device_map':self.device_map,
                                    'ddp':self.ddp,
                                    'base_model':self.default_model,
                                
                                    'data_path':self.train_data_path,
                                    'output_dir':default_path,
                                    'tokenizer_name':self.default_model,
                                    'load_in_8bit':True,
                                    'torch_dtype':torch.float16,
                                    })
        self.current_config.update({'gradient_accumulation_steps':self.gradient_accumulation_steps,
                                    'device_map':self.device_map,
                                    'ddp':self.ddp,
                                    'base_model':self.base_model,
                                    'data_path':self.train_data_path,
                                    'output_dir':self.output_dir,
                                    'tokenizer_name':self.base_model,
                                    'load_in_8bit':True,
                                    'torch_dtype':torch.float16,
                                    })
        
        self.Path_parameter = ['base_model','data_path','output_dir','prompt_template_name',]
        self.Train_parameter = ['batch_size','micro_batch_size','gradient_accumulation_steps','num_epochs','learning_rate', 'cutoff_len','val_set_size','train_on_inputs','add_eos_token','group_by_length','resume_from_checkpoint','ddp','warmup_steps','eval_steps','save_steps','save_total_limit',]
        self.Lora_parameter = ['lora_r','lora_alpha','lora_dropout','lora_target_modules']
        self.Record_parameter = ['wandb_project','wandb_run_name','wandb_watch','wandb_log_model',]
        self.Load_parameter = ['load_in_8bit','torch_dtype','device_map',]
        #self.Load_tokenizer_parameter
        
    def wandb_config(self):
        self.wandb_initiate = True
        os.environ["WANDB_API_KEY"] = 'ba872b4934f6ae8c68d8c63e3a03f51c0aa80762'
        wandb.init(project="text_spectral_analysis",
                   name = self.output_dir.split('/')[-2] + '/' + self.output_dir.split('/')[-1],
                   )
        return print('Wandb Initiated Successfully')
        

    def print_color(self, text):
        display(HTML(f'<font color="red">{text}</font>'))

    def print_config(self,):
        print('Current Configuration')
        print('--------------------------------------')
        for key in self.Path_parameter:
            text = f'{key}: {self.current_config[key]}'
            if self.current_config[key]!=self.default_config[key] or key not in self.default_config.keys():
                self.print_color(text)
            else:print(text)
        print('--------------------------------------')
        for key in self.Load_parameter:
            text = f'{key}: {self.current_config[key]}'
            if self.current_config[key]!=self.default_config[key] or key not in self.default_config.keys():
                self.print_color(text)
            else:print(text)
        print('--------------------------------------')
        for key in self.Lora_parameter:
            text = f'{key}: {self.current_config[key]}'
            if self.current_config[key]!=self.default_config[key] or key not in self.default_config.keys():
                self.print_color(text)
            else:print(text)
        print('--------------------------------------')
        for key in self.Train_parameter:
            text = f'{key}: {self.current_config[key]}'
            if self.current_config[key]!=self.default_config[key] or key not in self.default_config.keys():
                self.print_color(text)
            else:print(text)
        print('--------------------------------------')
        for key in self.Record_parameter:
            text = f'{key}: {self.current_config[key]}'
            if self.current_config[key]!=self.default_config[key] or key not in self.default_config.keys():
                self.print_color(text)
            else:print(text)
        print('--------------------------------------')
        for key,value in self.current_config.items():
            if key not in self.Path_parameter and key not in self.Load_parameter and key not in self.Lora_parameter and key not in self.Train_parameter and key not in self.Record_parameter:
                print(f'{key}: {self.current_config[key]}')
        return print('--------------------------------------')
    
    def load_template(self, prompt_template_path = None):
        if prompt_template_path is None:
            print(f'Default template : {self.prompt_template_name}')
            self.prompter = Prompter(self.prompt_template_name) # Loading Default Template
        else:
            print(f'Loading template : {prompt_template_path}')
            self.prompt_template_name = prompt_template_path
            self.prompter = Prompter(self.prompt_template_name) 
        return print('Template Loaded Successfully')
    
    def load_data(self):
        # Read train dataset
        if self.train_data_path.endswith(".json") or self.train_data_path.endswith(".jsonl"):  # 读取数据集
            print(self.train_data_path)
            self.train_data = load_dataset("json", data_files = self.train_data_path)
        else:
            self.train_data = load_dataset(self.train_data_path)
            print('Origin data loaded Successfully')
        return self.train_data

    def check_dict(self,data,key):
        value = [(i[0]['{}'.format(key)]) for i in data]
        return value, collections.Counter(value)

    from datasets import Dataset, DatasetDict

    def preprocess(self, data, validation_size=200):
    # 从 DatasetDict 中提取训练数据分割
        dataset = data['train']

    # 将数据转换为列表格式
        data_list = [dataset[i] for i in range(len(dataset))]

    # 手动分割数据
        valid_data_list = data_list[:validation_size]  # 取前200条作为验证集
        train_data_list = data_list[validation_size:]  # 剩余的作为训练集

    # 直接将列表数据转换为 Dataset
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data_list))
        valid_dataset = Dataset.from_pandas(pd.DataFrame(valid_data_list))

    # 将数据封装到 DatasetDict 中
        data_preprocessed = DatasetDict({
        'Train': train_dataset,
        'Valid': valid_dataset
        })

        print('Data preprocessing done.')
        return data_preprocessed


    def load_tokenizer(self,
                       tokenizer_name: Optional[str] = None, 
                       change_tokenizer: bool = False,
                       **token_kwargs,
                       ) -> AutoTokenizer:
        """
        Loading a specific tokenizer.
        Args:
            tokenizer_name (`str`, *optional*, defaults to `unsloth/llama-3-8b'`): 
                The character labels required for loading tokenizer through huggingface, which can be looked up on the huggingface and should also match the model you want to use.
            change_tokenizer (`bool`): 
                The key to change the tokenizer parameters.If you want to change the parameters, just set "change_tokenizer" to be True.
            token_kwargs (`Dict`):
                Parameters that can be changed include: "bos_token_id", "eos_token_id", "pad_token_id", "padding_side", etc.
        """
        if self.model:
            #print('Has model')
            if tokenizer_name != None:
                if tokenizer_name != self.base_model:
                    print(TypeError("The tokenizer_name and model_name must be the same. Loading in this way may have unpredictable consequences later in the process."))
                    self.tokenizer_name = self.base_model
                else:
                    self.tokenizer_name = self.base_model
            else:
                self.tokenizer_name = self.base_model
        else:
            #print('No model')
            if self.base_model == 'chat-gpt':
                self.tokenizer_name = self.base_model
            elif tokenizer_name != None:
                self.tokenizer_name = tokenizer_name
            else:self.tokenizer_name = self.default_model

        print(f'Loading Tokenizer : "{self.tokenizer_name}"')

        if self.tokenizer_name == 'chat-gpt': 
            pass
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name,token = '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',trust_remote_code=True)
            except Exception as e:
                print(e)
                Local_model_home = './MODEL/Model/meta-llama/Llama-2-7b-hf'    # Local_model_home should be changed to the local model save path.
                model_path = Local_model_home + self.tokenizer_name
                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(model_path,token = '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')
                except Exception as e:
                    print(e)
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path,token = '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')
                    except Exception as e:
                        raise

        if self.tokenizer_name != 'chat-gpt':
            print(f'Tokenizer: {self.tokenizer_name} Loaded Successfully\n')

        # Special positional parameter settings for tokens
        if change_tokenizer:
            print('Change config of the tokenizer.')
            if not token_kwargs:
                default_change_config = {'pad_token_id':(0),'padding_side':'left'}
                print('Loading the default tokenizer configuration.')
                self.tokenizer = self.change_parameter(self.tokenizer,default_change_config) # loading the default config or do not change the config of token
            else:
                self.tokenizer = self.change_parameter(self.tokenizer,token_kwargs)
        if self.tokenizer:
            print('\nBeginning token : {} , Beginning token id : {} \n'.format(self.tokenizer.bos_token,self.tokenizer.bos_token_id))
            print('Ending token : {} , Ending token id : {} \n'.format(self.tokenizer.eos_token,self.tokenizer.eos_token_id))
            print('Padding token : {} , Padding token id : {} \n'.format(self.tokenizer.pad_token,self.tokenizer.pad_token_id))
        
        return self.tokenizer

    def change_parameter(self,changed,parameter):
        log = {}
        for i in parameter.keys():
            print(i)
            if hasattr(changed,i): # check this attribute
                if getattr(changed,i)!= parameter[i]:
                    setattr(changed,i,parameter[i])      # change this attribute
                    log[i] = parameter[i]      # record this change
        df = pd.DataFrame([log.keys(), log.values()]).T
        df.columns = ['Parameter', 'Value']
        print(df)  
        return changed

    def tokenize(self,prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        # 在文本末端添加一位休止符
        # 注意力掩膜同样添加一位，因为这个位置需要训练
        result = self.tokenizer( # LlamaTokenizer → PreTrainedTokenizer → PreTrainedTokenizerBase → 1364行处
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self,data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["Instruction"],
            data_point["Question"],
            data_point["Output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt) # 添加休止符
        if not self.train_on_inputs: # 如果为 False，则屏蔽损失中的输入。默认设置train_on_inputs取True
            user_prompt = self.prompter.generate_prompt(
                data_point["Instruction"], data_point["Question"]
            )
            tokenized_user_prompt = self.tokenize(   # 不增加休止符
                user_prompt, add_eos_token=self.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if self.add_eos_token:
                user_prompt_len -= 1  # 实际的文本长度，去除了最后一位休止符

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably     
            # 这里的作用是消去instruction和input的所有影响，只对output部分进行训练，因为标签为负。对应初始化条件中train_on_inputs 为False
        return tokenized_full_prompt
    
    def map_dataset(self,dataset):
        if self.val_set_size > 0:
            self.train_data = (
                dataset["Train"].shuffle(seed=32).map(self.generate_and_tokenize_prompt)
            )
            self.val_data = (
                dataset["Valid"].shuffle(seed=32).map(self.generate_and_tokenize_prompt)
            )
        else:
            self.train_data = dataset["Train"].shuffle(seed=32).map(self.generate_and_tokenize_prompt)
            self.val_data = None   # 不设置验证集
        print('Data Mapped Successfully')
        if self.val_set_size > 0:
            return self.train_data, self.val_data
        else:return self.train_data

    def load_model(self, 
                   model_name: Optional[str] = None, 
                   model_config: Optional[dict] = None,
                   ) -> Optional[AutoModelForCausalLM]:
        """
        Loading specific model and possible lora adaption.
        Args:
            model_name (`str`, *optional*, defaults to `'meta-llama/Meta-Llama-3-8B'`): 
                The character labels required for loading tokenizer through huggingface, which can be looked up on the huggingface and should also match the model you want to use.
            lora_name (`str`, *optional*): 
                The key to change the tokenizer parameters.If you want to change the parameters, just set "change_tokenizer" to be True.
        """
        # Clear the cache of model.
        if self.tokenizer:
            #print('has tokenizer')
            if model_name != None:
                if model_name != self.tokenizer_name:
                    print(TypeError("The model_name and tokenizer_name must be the same. Loading in this way may have unpredictable consequences later in the process."))
                    self.base_model = self.tokenizer_name
                else:
                    self.base_model = self.tokenizer_name
            else:
                self.base_model = self.tokenizer_name
        else:
            #print('no tokenizer')
            if self.tokenizer_name == 'chat-gpt':
                self.base_model = self.tokenizer_name
            elif model_name != None:
                self.base_model = model_name
            else:self.base_model = self.default_model

        print(f'Loading Model : "{self.base_model}"')

        # update_config
        if model_config:
            self.current_config.update(model_config)
        input_config = {k: v for k, v in self.current_config.items() if k in self.Load_parameter}
        print(input_config)
        # analytic model
        if self.base_model == self.default_model:
            self.model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            token = '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',
            **input_config)

        elif self.base_model == 'chat-gpt':
            print('Launch chat-gpt.')

        else:
            try: 
                self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                token = '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',
                **input_config)
            except Exception as e:
                print(e)
                Local_model_home = './MODEL/Model/meta-llama/Llama-2-7b-hf'    # Local_model_home should be changed to the local model save path.
                model_path = Local_model_home + self.base_model
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    token ='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',
                    **input_config)
                except Exception as e:
                    print(e)
                    raise('Cannot load this model.Please check the path of model.')
        print('Model: {} --- Loaded Successfully'.format(self.base_model))

        return self.model

    def model_prepare(self,
                      lora_config = None,
                      resume_from_checkpoint = None):
        self.model = prepare_model_for_int8_training(self.model)  
        if lora_config:
            self.current_config.update(lora_config)
        input_config = {k: v for k, v in self.current_config.items() if k in self.Lora_parameter}
        input_config["r"] = input_config.pop("lora_r")
        input_config["target_modules"] = input_config.pop("lora_target_modules")
        config = LoraConfig(
            bias="none",
            task_type="CAUSAL_LM",
            **input_config,
        )
        #print(config)
        self.model = get_peft_model(self.model, config)

        # Check the available weights and load them   # 检查是否从之前的训练处重新开始
        if resume_from_checkpoint != None:self.resume_from_checkpoint = resume_from_checkpoint
        if self.resume_from_checkpoint:   
            checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
            )  # 从Full checkpoint开始
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.safetensors"
                )  # 从only LoRA model开始 - LoRA config above has to fit
                resume_from_checkpoint = (
                    False  # So the trainer won't try loading its state
                )
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(self.model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")

        if not self.ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            self.model.is_parallelizable = True
            self.model.model_parallel = True  
        self.model.config.use_cache = False
        self.model.print_trainable_parameters()    # 打印模型中可训练参数量        
        print('Model Prepared Successfully')
        return config

    def compute_metrics(self):
        pass

    def Hyperparametric_search(self):
        pass

    def train(self,train_data=None, val_data=None, output_dir=None, My_trainer=None, **train_config):
        # sub_path = 'G:/make_question/new_verson/Train/Output' + '/{}'.format(datetime.datetime.now().strftime("%H-%M"))
        # if self.output_dir is None:
        #     current_datetime = datetime.datetime.now()
        #     current_date = current_datetime.strftime("%Y-%m-%d/")
        #     current_time = current_datetime.strftime("%H-%M")
        #     self.output_dir = os.path.join('./Train/Output/', current_date) + current_time
        #     output_dir = self.output_dir
        # else: output_dir = self.output_dir
        if output_dir:
            self.output_dir=output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if train_config:
            self.current_config.update(train_config)
        input_config = {k: v for k, v in self.current_config.items() if k in self.Train_parameter}
        run_name = self.output_dir.split('/')[-2] + '/' + self.output_dir.split('/')[-1]
        self.Arguments = transformers.TrainingArguments(  
            # Parameters subject to change
            per_device_train_batch_size = input_config.get('micro_batch_size',4),
            gradient_accumulation_steps = input_config.get('gradient_accumulation_steps',32),
            warmup_steps = input_config.get("warmup_steps", 100),
            num_train_epochs = input_config.get("num_epochs", 5),
            learning_rate = input_config.get("learning_rate", 3e-4),
            eval_steps = input_config.get("eval_steps", 50) if self.val_set_size > 0 else None,
            save_steps = input_config.get("save_steps", 50),
            save_total_limit = input_config.get("save_total_limit", 5),
            output_dir = self.output_dir,

            # Parameters keep default
            fp16 = True,
            logging_steps = 10,
            optim = "adamw_torch",
            evaluation_strategy = "steps" if self.val_set_size > 0 else "no",
            save_strategy = "steps",
            load_best_model_at_end = True if self.val_set_size > 0 else False,
            ddp_find_unused_parameters = False if self.ddp else None,
            group_by_length = self.group_by_length,
            report_to = 'wandb' if self.wandb_initiate else 'none',
            run_name = run_name if self.wandb_initiate else None,
            #label_names= ['weight_label'] if My_trainer else None,
            remove_unused_columns = False if My_trainer else True,
            #report_to = "wandb" if self.use_wandb else None,
            #run_name = self.wandb_run_name if self.use_wandb else None,
        )
        self.print_config()
        # 打印出Arguments中从change_parameter中获取的参数
        # if not train_config:
        #     print('Keep default hyper-parameter.')
        #     self.print_config()
        # else:
        #     change_list = [i for i in change_parameter if i in Arguments.__dict__.keys()]
        #     print('Change hyper-parameter:\n')
        #     for key in change_list:
        #         print(key, ' : ', change_parameter[key])
        #     print('\n')
        #     self.print_config()

        if not My_trainer:
            # 加载原始版本的训练器
            self.trainer = transformers.Trainer( # evaluate 在2982行
            model = self.model,
            train_dataset = train_data if train_data != None else self.train_data,
            eval_dataset = val_data if val_data != None else self.val_data,
            args = self.Arguments,
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        #compute_loss = self.compute_metrics
        # #compute_metrics = 
            )
        elif My_trainer:
            # 加载自定义的训练器
            self.trainer = My_trainer(
            model = self.model,
            train_dataset = train_data if train_data != None else self.train_data,
            eval_dataset = val_data if val_data != None else self.val_data,
            args = self.Arguments,
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            )

        return self.Arguments
    
    def predict(self):
        pass
        # print('Prediction Started')
        # self.predictions = self.model.predict(self.test_data)
        # print('Prediction Done')

    def new_loss(self):
        pass

