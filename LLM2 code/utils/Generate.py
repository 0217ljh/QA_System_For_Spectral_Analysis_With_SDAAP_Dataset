import os
import sys
import fire
import torch
import evaluate
import transformers
import numpy as np
import torch.nn as nn
import pandas as pd
import datetime
import wandb
import bitsandbytes
import collections
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.prompter import Prompter
from typing import List, Dict, Optional, Tuple, Union
from datasets import load_dataset, Dataset, DatasetDict
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)



class Generate():
    def __init__(self, **Configs):
        # Initialisation Configuration
        self.B_data: Optional[Dataset] = None
        self.O_data: Optional[Dataset] = None
        self.predict_result: Optional[Dataset] = None
        self.predict_data: Optional[DatasetDict] = None
        self.data_preprocess: Optional[DatasetDict] = None
        self.predict_data_path: Optional[str] = None

        self.tokenizer:Optional[AutoTokenizer] = None
        self.tokenizer_name: Optional[str] = None

        self.prompt_template: Optional[str] = None
        self.prompter: Optional[Prompter] = None

        self.model: Optional[AutoModelForCausalLM] = None
        self.model_name: Optional[str] = None
        self.lora_path: Optional[str] = None
        self.output_dir = None   

        self.default_model_tokenizer: str = 'meta-llama/Meta-Llama-3-8B'
        self.add_eos_token: Optional[bool] = None
        self.generation_config: Optional[GenerationConfig] = None

        for key, value in Configs.items():
             setattr(self, str(key), value)
         
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
    
    def wandb_config(self,
                     project_name: Optional[str] = None,
                     experiment_name: Optional[str] = None,
                     notes: Optional[str] = None,
                     ):
        os.environ["WANDB_API_KEY"] = 'your wandb key'
        #os.environ["WANDB_WATCH"] = wandb_watch
        #os.environ["WANDB_LOG_MODEL"] = wandb_log_model
        ROOT_DIR = os.path.dirname(os.path.realpath('__file__'))
        ROOT_DIR = ROOT_DIR[:ROOT_DIR.find('new_verson') + len('new_verson')]
        wandb_save_path = ROOT_DIR + "/wandb"

        wandb.init(project=project_name, 
                   entity="ljhsysu46", 
                   name=experiment_name,
                   notes=notes,
                   dir=wandb_save_path,
                   #job_type='generate',
                   #config='all',
                   #reinit=True
                   )
        #wandb.config.update(self.__dict__)
        #wandb.log({'wandb_config': wandb.config})
        return print('Wandb config Done')
    
    def print_config(self):
        """
        Prints the configuration parameters used for inference.
        """
        print('Current Configuration:')
        for key, value in self.__dict__.items():
            if key =='train_data' or key == 'test_data' or key == 'val_data':
                 pass
            else:print(key, ' : ', value)
        return print('Print Done')
    
    def load_template(self, 
                      prompt_template_path: Optional[str] = None,
                      verbose: bool = False,
                      ) -> Prompter:
        """
        Processing and filtering the raw data to obtain a subset of data used to validate the model's capabilities in inference.
        Args:
            prompt_template_path (`DatasetDict`, *optional*, defaults to 'alpaca'): 
               The path for loading template files. Templates for supplemental prompts that can be easily customized.
            verbose (`bool`, defaults to `False`):  
                Whether to print the template description.
        """  
        if prompt_template_path is None:
            self.prompt_template = 'alpaca' # Default template
            print(f'Default template : {self.prompt_template}')
            self.prompter = Prompter(verbose=verbose) # Loading Default Template
        else:
            print(f'Loading template : {prompt_template_path}')
            self.prompt_template = prompt_template_path
            self.prompter = Prompter(self.prompt_template,verbose=verbose) # Loading Template
        print('Template Loaded Successfully')
        return self.prompter
    
    def load_data(self,
                  predict_data_path: str = '',
                  ) -> DatasetDict :
        """
        Prints the configuration parameters used for inference.
        Args:
            predict_data_path (`str`): Storage path for the dataset to be predicted.
        """
        if len(predict_data_path) == 0:
            raise TypeError("The path of predict data can't be null. Please enter the correct predictive data path.")
        else:self.predict_data_path = predict_data_path
        # Read train dataset
        print(f'Loading data: "{self.predict_data_path}"')
        if self.predict_data_path.endswith(".json") or self.predict_data_path.endswith(".jsonl"):  # 读取数据集
            try:
                self.predict_data = load_dataset("json", data_files = self.predict_data_path)
            except FileNotFoundError as e:
                raise
        else:
            try:
                self.predict_data = load_dataset(self.predict_data_path)
            except FileNotFoundError as e:
                print('except:', e)
                raise
        print('Predict data loaded successfully')
        return self.predict_data

    def check_dict(self,data,key):
        value = [(i[0]['{}'.format(key)]) for i in data]
        return value, collections.Counter(value)
    
    def load_tokenizer(self,
                       tokenizer_name: Optional[str] = None, 
                       change_tokenizer: bool = False,
                       **token_kwargs,
                       ) -> AutoTokenizer:
        """
        Loading a specific tokenizer.
        Args:
            tokenizer_name (`str`, *optional*, defaults to `"meta-llama/Meta-Llama-3-8B'`): 
                The character labels required for loading tokenizer through huggingface, which can be looked up on the huggingface and should also match the model you want to use.
            change_tokenizer (`bool`): 
                The key to change the tokenizer parameters.If you want to change the parameters, just set "change_tokenizer" to be True.
            token_kwargs (`Dict`):
                Parameters that can be changed include: "bos_token_id", "eos_token_id", "pad_token_id", "padding_side", etc.
        """
        if self.model:
            #print('Has model')
            if tokenizer_name != None:
                if tokenizer_name != self.model_name:
                    print(TypeError("The tokenizer_name and model_name must be the same. Loading in this way may have unpredictable consequences later in the process."))
                    self.tokenizer_name = self.model_name
                else:
                    self.tokenizer_name = self.model_name
            else:
                self.tokenizer_name = self.model_name
        else:
            #print('No model')
            if self.model_name == 'chat-gpt':
                self.tokenizer_name = self.model_name
            elif tokenizer_name != None:
                self.tokenizer_name = tokenizer_name
            else:self.tokenizer_name = self.default_model_tokenizer

        print(f'Loading Tokenizer : "{self.tokenizer_name}"')

        if self.tokenizer_name == 'chat-gpt': 
            pass
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name,token = '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',trust_remote_code=True)
            except Exception as e:
                print(e)
                Local_model_home = 'F:/Model/Local_model_home/'    # Local_model_home should be changed to the local model save path.
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
                if self.token_config:
                    print('Loading the default tokenizer configuration.')
                    self.tokenizer = self.change_parameter(self.tokenizer,self.token_config) # loading the default config or do not change the config of token
                else:
                    raise print('No default tokenizer configuration.')
            else:
                tokenizer_change_log = {}
                self.tokenizer = self.change_parameter(self.tokenizer,token_kwargs)
                # for i in token_kwargs.keys():
                #     if hasattr(self.tokenizer,i): # check this attribute
                #         setattr(self.tokenizer,i,token_kwargs[i])      # change this attribute
                #         tokenizer_change_log[i] = token_kwargs[i]      # record this change
                # df = pd.DataFrame([tokenizer_change_log.keys(), tokenizer_change_log.values()]).T
                # df.columns = ['Parameter', 'Value']
                # print(df)
        if self.tokenizer:
            print('\nBeginning token : {} , Beginning token id : {} \n'.format(self.tokenizer.bos_token,self.tokenizer.bos_token_id))
            print('Ending token : {} , Ending token id : {} \n'.format(self.tokenizer.eos_token,self.tokenizer.eos_token_id))
            print('Padding token : {} , Padding token id : {} \n'.format(self.tokenizer.pad_token,self.tokenizer.pad_token_id))
        
        return self.tokenizer

    def change_parameter(self,changed,parameter):
        log = {}
        for i in parameter.keys():
            if hasattr(changed,i): # check this attribute
                if getattr(changed,i)!= parameter[i]:
                    setattr(changed,i,parameter[i])      # change this attribute
                    log[i] = parameter[i]      # record this change
        df = pd.DataFrame([log.keys(), log.values()]).T
        df.columns = ['Parameter', 'Value']
        print(df)  
        return changed

    def tokenize(self,
                 prompt: str, 
                 ) -> Dict:
        """
        Tokenizerize the samples, add 'inputs_id','attention_mask' and 'labels' to the sample.
        Args:
            prompt (`str`): 
               Text data to be processed.
            
        """ 

        result = self.tokenizer( # LlamaTokenizer → PreTrainedTokenizer → PreTrainedTokenizerBase → 1364行处
            prompt,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.max_tokens
            and self.add_eos_token
        ): # add eos_token_id 
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1) # cover the eos_token_id

        result["labels"] = result["input_ids"].copy() # To be predicted in training

        return result

    def generate_and_tokenize_prompt(self,
                                     data_sample:Dict,
                                     ) -> Dict:
        """
        For processing samples one by one.
        Args:
            data_sample (`Dict`): 
               The sample must contain the 'instruction' and 'input' label.
        """ 
        full_prompt = self.prompter.generate_prompt(
            data_sample["Instruction"],
            data_sample["Question"],
        )
        if self.tokenizer:
            tokenized_full_prompt = self.tokenize(full_prompt) # add eos_token_id 
        else:tokenized_full_prompt = {'template_instruction' : full_prompt}
        return tokenized_full_prompt 
    
    def Preprocess(self, 
                   data: Optional[DatasetDict] = None, 
                   ) -> DatasetDict:
        """
        Processing and filtering the raw data to obtain a subset of data used to validate the model's capabilities in inference.
        Args:
            data (`DatasetDict`, *optional*, defaults to `self.predict_data`): 
               The DatasetDict must contain the 'train' label, whose subset of Datset must contain the 'Question Index' and 'TYPE' labels.
        """           
        if not data:
            predict_data = self.predict_data
            print('Processing Data.') 
        else: 
            predict_data = data
            print('Processing of new input data.')
        choose_first_question = predict_data['train']
        self.data_preprocess = DatasetDict({
        'Both Train and Test' : choose_first_question,
        'Only Test' : choose_first_question
                     })
        print('Data Preprocess Done')
        return self.data_preprocess

    def map_dataset(self,
                    dataset: DatasetDict,
                    max_tokens: Optional[int] = None,
                    add_eos_token = True,
                    ) -> Dataset:
        """
        Map processing of dataset based on tokenizer and template. The tokenizer and template must be initialized before.
        Args:
            dataset (`DatasetDict`,defaults to `self.data_preprocess`): 
               The DatasetDict must contain the 'Both Train and Test' and 'Only Test' label.
            add_eos_token (`bool`, *optional*, defaults to `True`):
                Whether to add an 'add_eos_token_id' at the end of 'inputs_id','attention_mask' and 'labels'.Note that adding 'add_eos_token' during inference will cause the model to not answer properly and needs to be set to False. Just set 'add_eos_token' to be True during training.
        """ 
        if not dataset:
            if not self.data_preprocess:
                raise TypeError("The dataset can't be null. Please enter the correct dataset.")
            dataset = self.data_preprocess
        if self.tokenizer:
            self.add_eos_token = add_eos_token
            self.max_tokens = 256 if max_tokens != None else 256
            print('Start mapping data.')
            if self.add_eos_token:
                print('Add eos token in the end of inputs_id.')
        self.B_data = (
            dataset["Both Train and Test"].map(self.generate_and_tokenize_prompt)
        )
        self.O_data = (
            dataset["Only Test"].map(self.generate_and_tokenize_prompt)
        )
        
        print('Data Mapped Successfully')
        
        return self.B_data, self.O_data
       
    def load_model(self, 
                   model_name: Optional[str] = None, 
                   lora_name: Optional[str] = None,
                   lora_config: Optional[dict] = None,
                   model_config: Optional[dict] = None,
                   ) -> Union[ AutoModelForCausalLM, PeftModel]:
        """
        Loading specific model and possible lora adaption.
        Args:
            model_name (`str`, *optional*, defaults to `"meta-llama/Meta-Llama-3-8B'`): 
                The character labels required for loading tokenizer through huggingface, which can be looked up on the huggingface and should also match the model you want to use.
            lora_name (`str`, *optional*): 
                The key to change the tokenizer parameters.If you want to change the parameters, just set "change_tokenizer" to be True.
        """
        # Clear the cache of model.
        if self.tokenizer:
            print('has tokenizer')
            if model_name != None:
                if model_name != self.tokenizer_name:
                    print(TypeError("The model_name and tokenizer_name must be the same. Loading in this way may have unpredictable consequences later in the process."))
                    self.model_name = self.tokenizer_name
                else:
                    self.model_name = self.tokenizer_name
            else:
                self.model_name = self.tokenizer_name
        else:
            print('no tokenizer')
            if self.tokenizer_name == 'chat-gpt':
                self.model_name = self.tokenizer_name
            elif model_name != None:
                self.model_name = model_name
            else:self.model_name = self.default_model_tokenizer

        print(f'Loading Model : "{self.model_name}"')

        # analytic model
        if self.model_name == self.default_model_tokenizer:
            self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            **self.model_config if not model_config else model_config,)

        elif self.model_name == 'chat-gpt':
            print('Launch chat-gpt.')

        else:
            try: 
                self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **self.model_config if not model_config else model_config)
            except Exception as e:
                print(e)
                Local_model_home = 'F:/Model/Local_model_home/'    # Local_model_home should be changed to the local model save path.
                model_path = Local_model_home + self.model_name
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **self.model_config if not model_config else model_config)
                except Exception as e:
                    print(e)
                    raise('Cannot load this model.Please check the path of model.')
        print('Model: {} --- Loaded Successfully'.format(self.model_name))

        # analytic lora
        if lora_name == None:
            print('Without lora adaption.')
        else:
            self.lora_path = lora_name
            self.loading_lora_adpation(self.lora_path)
        return self.model
    
    def loading_lora_adpation(self,
                              lora_name:str,
                              ):
        """
        Loading lora adaption.
        Args:
            lora_path (`str`): 
                The character labels required for loading tokenizer through huggingface, which can be looked up on the huggingface and should also match the model you want to use.
        """
        try:
            self.model = PeftModel.from_pretrained(
            self.model,
            lora_name,
            torch_dtype=torch.float16,
            )
        except Exception as e:
            print('except:', e)
            raise      
        return print('Lora: {}\nLoaded Successfully'.format(lora_name))

    def Parameter_setting(self,
                          generate_config : Optional[dict] = None,
                          ):
        # 1.token analysis 2.Inference parameter 3.save path
        # 1.token and model
        #self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        #self.model.config.bos_token_id = 1
        #self.model.config.eos_token_id = 2

        # 2.Inference paramter
        self.generation_config = GenerationConfig( ## Can try **kwargs.
            **self.generate_config if not generate_config else generate_config,
            )
        print('Parameter setting Done')
        return self.generation_config

    def new_loss(self):
        pass

    def compute_metrics(self):
        pass

    def Hyperparametric_search(self):
        pass

    def train(self,train_data=None, val_data=None, **change_parameter):
        sub_path = self.output_dir + '/{}'.format(datetime.datetime.now().strftime("%H-%M"))
        Arguments = transformers.TrainingArguments(  
            # Parameters subject to change
            per_device_train_batch_size = self.micro_batch_size,
            gradient_accumulation_steps = self.gradient_accumulation_steps,
            warmup_steps = change_parameter.get("warmup_steps", 100),
            num_train_epochs = change_parameter.get("num_train_epochs", 3),
            learning_rate = change_parameter.get("learning_rate", 3e-4),
            eval_steps = change_parameter.get("eval_steps", 10) if self.val_set_size > 0 else None,
            save_steps = change_parameter.get("save_steps", 10),
            save_total_limit = change_parameter.get("save_total_limit", 5),
            output_dir = sub_path,

            # Parameters keep default
            fp16 = True,
            logging_steps = 10,
            optim = "adamw_torch",
            evaluation_strategy = "steps" if self.val_set_size > 0 else "no",
            save_strategy = "steps",
            load_best_model_at_end = True if self.val_set_size > 0 else False,
            ddp_find_unused_parameters = False if self.ddp else None,
            group_by_length = self.group_by_length,
            #report_to = "wandb" if self.use_wandb else None,
            #run_name = self.wandb_run_name if self.use_wandb else None,
        )
        # 打印出Arguments中从change_parameter中获取的参数
        if not change_parameter:
            print('Keep default hyper-parameter.')
        else:
            change_list = [i for i in change_parameter if i in Arguments.__dict__.keys()]
            print('Change hyper-parameter:\n')
            for key in change_list:
                print(key, ' : ', change_parameter[key])

        self.trainer = transformers.Trainer( # evaluate 在2982行
        model = self.model,
        train_dataset = train_data if train_data != None else self.B_data,
        eval_dataset = val_data if val_data != None else self.O_data,
        args = Arguments,
        data_collator=transformers.DataCollatorForSeq2Seq(
             self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # #compute_metrics = 
        )
        return Arguments
    
    def predict_circulate(self, 
                          data:Dataset, 
                          is_shut_down:bool = False,
                          shut_down_index: Optional[int] = None,
                          ) -> Dataset:
        """
        Prediction of input data.
        Args:
            data (`Dataset`,): 
                Data for inference using models.
            config (`GenerationConfig`, *optional*, defaults to None): 
                Hyperparameter configurations invoked during model inference.
            is_shut_down (`bool`, *optional*, defaults to False):
                Whether to set the abort position.For quick monitoring of predicted results.
            shut_down_index (`int`, *optional*, defaults to None):
                The abort position.
        """
        if data == None:raise('Data cannot be None.')
        columns = data.column_names
        if self.model:
            self.model.eval()
            columns_to_keep = ['Instruction', 'Question','Knowledge','Output','input_ids']  # Only keep this special columns.
        else:
            columns_to_keep = ['Instruction','Question', 'Knowledge','Output','template_instruction']
        columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
        issues_dataset = data.remove_columns(columns_to_remove)
        Fill = [None]*len(issues_dataset)
            
        for i in tqdm(range(len(issues_dataset))):
            if is_shut_down: # Setting the abort position
                if i > shut_down_index:break 

            if self.model:
                if self.add_eos_token:
                    input_ids = torch.tensor(issues_dataset[i]['input_ids'][:-1]).view(1,-1).to(self.device) # remove the Ending token id
                else:input_ids = torch.tensor(issues_dataset[i]['input_ids']).view(1,-1).to(self.device)  
                output = self.predict_sample(input_ids)
            else:
                if i ==0:key_index=None
                output,key_index = self.predict_gpt(issues_dataset[i]['template_instruction'],key_index)
            Fill[i] = output
        issues_dataset = issues_dataset.add_column('model_response', Fill)
        self.predict_result = issues_dataset
    
        return issues_dataset
       
    def predict_gpt(self,prompt,key_index=None):
        from langchain.chat_models import ChatOpenAI                   
        from langchain.schema import HumanMessage,SystemMessage
        KEY = ['OpenAI_Key1',
               'OpenAI_Key2'
       ]
        if not key_index:
            key_index = 0
        os.environ['OPENAI_API_KEY']= KEY[key_index]
        llm =ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1, max_tokens = 256)
        human = HumanMessage(content=prompt)
        messages1 = [human]
        try:
            result = llm(messages1)
        except:
            key_index = key_index + 1
            os.environ['OPENAI_API_KEY']= KEY[key_index]
            llm =ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1, max_tokens = 256)
            result = llm(messages1)
        string_result = dict(result)['content']
        return string_result,key_index

    def predict_sample(self,input_ids):
        with torch.no_grad():
            generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            )
        s = generation_output[0]
        output = self.tokenizer.decode(s)
        #print(output)
        Output = self.prompter.get_response(output)
        return Output
    
    def save_results(self, results:Dataset, output_dir=None):
        # 3.save path
        if output_dir:
            finnal_dir = output_dir
            # 检测save_path是否存在
        elif self.output_dir != None:
            finnal_dir = self.output_dir
        else:
            ROOT_DIR = os.path.dirname(os.path.realpath('__file__'))
            ROOT_DIR = ROOT_DIR[:ROOT_DIR.find('new_verson') + len('new_verson')]
            project_path = ROOT_DIR + "/Generate/Output/" # Absolute path under the project
            if self.lora_path:
                save_path = project_path + self.model_name.replace('/','_') + '@' + self.lora_path.replace('/','_')
            else:save_path = project_path + self.model_name.replace('/','_')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            current_datetime = datetime.datetime.now()
            current_date = current_datetime.strftime("%Y-%m-%d")
            finnal_dir = os.path.join(save_path, current_date) 
        if not os.path.exists(finnal_dir):
            os.makedirs(finnal_dir)
        print(f'Saving path: {finnal_dir}') 
        self.output_dir = finnal_dir
        results.to_json(finnal_dir + '/Only/{}.json'.format(datetime.datetime.now().strftime("%H_%M_%S")),indent=4,)
        return print('Results Saved Successfully')
