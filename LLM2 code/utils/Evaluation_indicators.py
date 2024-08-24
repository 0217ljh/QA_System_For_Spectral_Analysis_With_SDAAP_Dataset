import evaluate
import re
import json
import numpy as np
import collections
from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Optional, Tuple, Union

class Evaluation_indicators:
    def __init__(self,):
        self.dataset = None

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
                self.dataset = load_dataset("json", data_files = self.predict_data_path)['train']
            except FileNotFoundError as e:
                raise
        else:
            try:
                self.dataset = load_dataset(self.predict_data_path)['train']
            except FileNotFoundError as e:
                print('except:', e)
                raise
        print('Predict data loaded successfully')
        return self.dataset

    def run_evaluate(self,
                     evaluate_set=None,
                     choose_method:list=None):
        Record = {}
        if not evaluate_set:
            evaluate_set = self.dataset
        if not choose_method:
            choose_method = ['bleu','rouge','meteor','bert_score','bleurt']
            for i in choose_method:
                Record.update(self.call_fun_by_str(i,evaluate_set))
        else:
            for i in choose_method:
                Record.update(self.call_fun_by_str(i,evaluate_set))
        return Record
    
    def call_fun_by_str(self,
                        fun_str:str,
                        evaluate_set,
                        ) -> Dict:
        c = fun_str
        c_evaluator = c.upper()
        fun_loading = getattr(self, c, None)
        if not hasattr(self,c_evaluator):
            Evaluator = fun_loading()
        else:
            Evaluator = getattr(self, c_evaluator, None)
        fun_loading = getattr(self, 'run_'+c, None)
        score = fun_loading(Evaluator,evaluate_set)
        record = self.save_items(score,c)
        return record
    
    def run_bleu(self,
                 Evaluator,
                 dataset,
                 ) -> Union[float,List,Dict]:
        bleu_score = []
        bleu_all = []
        Length = len(self.dataset) if not dataset else len(dataset)
        for i in range(Length):
            #if i >5:break
            predictions = [self.dataset[i]['model_response']] if not dataset else [dataset[i]['model_response']]
            references = [self.dataset[i]['Output']] if not dataset else [dataset[i]['Output']]
            result = Evaluator.compute(predictions=predictions, references=references)
            bleu_score.append(result['bleu'])
            bleu_all.append(result)
        return {'bleu_score':np.mean(bleu_score)}

    def run_rouge(self,
                  Evaluator,
                  dataset,
                 ) -> Union[float,List,Dict]:
        rouge_all = []
        rouge1_score = []
        rouge2_score = []
        rougeL_score = []
        Length = len(self.dataset) if not dataset else len(dataset)
        for i in range(Length):
            #if i >5:break
            predictions = [self.dataset[i]['model_response']] if not dataset else [dataset[i]['model_response']]
            references = [self.dataset[i]['Output']] if not dataset else [dataset[i]['Output']]
            result = Evaluator.compute(predictions=predictions, 
                                references=references,
                                rouge_types=["rouge1", "rouge2", "rougeL",],
                                )
            rouge1_score.append(result['rouge1'])
            rouge2_score.append(result['rouge2'])
            rougeL_score.append(result['rougeL'])
            rouge_all.append(result)
        return {'rouge1':np.mean(rouge1_score),'rouge2':np.mean(rouge2_score),'rougeL':np.mean(rougeL_score)}
        
    def run_bert_score(self,
                       Evaluator,
                       dataset,
                       ) -> Union[float,List,Dict]:
        BERT_all = []
        precision_score = []
        recall_score = []
        f1_score = []
        Length = len(self.dataset) if not dataset else len(dataset)
        for i in range(Length):
            #if i >5:break
            predictions = [self.dataset[i]['model_response']] if not dataset else [dataset[i]['model_response']]
            references = [self.dataset[i]['Output']] if not dataset else [dataset[i]['Output']]
            result = Evaluator.compute(predictions=predictions, references=references, 
                                    lang = 'en',
                                    #model_type="distilbert-base-uncased"
                                    )
            precision_score.append(result['precision'])
            recall_score.append(result['recall'])
            f1_score.append(result['f1'])
            BERT_all.append(result)
        return {'precision_score':np.mean(precision_score),'recall_score':np.mean(recall_score),'f1_score':np.mean(f1_score)}

    def run_bleurt(self,
                   Evaluator,
                   dataset,):
        bleurt_score = []
        bleurt_all = []
        Length = len(self.dataset) if not dataset else len(dataset)
        for i in range(Length):
            #if i >5:break
            predictions = [self.dataset[i]['model_response']] if not dataset else [dataset[i]['model_response']]
            references = [self.dataset[i]['Output']] if not dataset else [dataset[i]['Output']]
            result = Evaluator.compute(predictions=predictions, references=references,
            )['scores']
            bleurt_score.append(result)
        return {'bleurt_score_aver':np.mean(bleurt_score)}

    def run_meteor(self,
                   Evaluator,
                   dataset,
                   ) -> Union[float,List,Dict]:
        meteor_score = []
        Length = len(self.dataset) if not dataset else len(dataset)
        for i in range(Length):
            #if i >5:break
            predictions = [self.dataset[i]['model_response']] if not dataset else [dataset[i]['model_response']]
            references = [self.dataset[i]['Output']] if not dataset else [dataset[i]['Output']]
            result = Evaluator.compute(predictions=predictions, references=references,
            )['meteor']
            meteor_score.append(result)
        return {'meteor_score_aver':np.mean(meteor_score)}

    def bleu(self):
        self.BLEU = evaluate.load("bleu")
        print('bleu loaded')
        return self.BLEU
    
    def rouge(self):
        self.ROUGE = evaluate.load("rouge")
        print('rouge loaded')
        return self.ROUGE

    def bert_score(self):
        self.BERT_SCORE = evaluate.load("bertscore")
        print('bert_score loaded')
        return self.BERT_SCORE

    def bleurt(self):
        self.BLEURT = evaluate.load("bleurt",
                           'bleurt-large-512',
                           module_type="metric",
                           )
        print('bleurt loaded')
        return self.BLEURT

    def meteor(self):
        self.METEOR= evaluate.load('meteor')
        print('meteor loaded')
        return self.METEOR

    def run_more_exact_match(self,
                     Evaluator,
                        dataset,
                        ) -> dict:
        is_answer_correct = []
        if not dataset:
            dataset = self.dataset
        kind_key = collections.Counter(dataset['Spectral Kind'])
        save_answer = {i:[] for i in kind_key}
        for i in dataset:
            answer = self.sample_match(i,Evaluator)
            for j in kind_key:
                if i['Spectral Kind'] ==  j:
                    save_answer[j].append(i)
                    save_answer[j][-1]['answer'] = answer
                    break
            is_answer_correct.append(answer)
        Output = {}
        correct_number = [i for i, x in enumerate(is_answer_correct) if x == True]
        Output['correct_number'] = len(correct_number)
        for key,value in save_answer.items():
            baifenbi = len([i for i in value if i['answer'] == True])/len(value)
            Output[key] = [baifenbi,len([i for i in value if i['answer'] == True]),len(value)]
        return Output

    def run_exact_match(self,
                   Evaluator,
                   dataset,
                   ) -> dict:
        is_answer_correct = []
        if not dataset:
            for i in self.dataset:
                answer = self.sample_match(i,Evaluator)
                is_answer_correct.append(answer)
        else:
            for i in dataset:
                answer = self.sample_match(i,Evaluator)
                is_answer_correct.append(answer)
        correct_index = [i for i, x in enumerate(is_answer_correct) if x == True]
        accuary = len(correct_index)/len(is_answer_correct)
        return {'accuary':accuary}

    def exact_match(self):
        self.EXACT_MATCH={
        '1':['uv' , 'ultravio'], 
        '2':['vis','visible'], 
        '3.1':['nir','near-infrared','near infrared','near infra-red','SWIR'], 
        '3.2':['mir','mid-infrared','mid-ir','mid infrared','mid-DRIFTS','midinfrared'], 
        '3.3':['ftir','Fourier transform infrared','ft-ir' ,'Fourier-transform infrared'], 
        '3.4':['long-wave infrared','long wave infrared','lwir','lir'],
        '4':['raman','SERS'], 
        '5':['LIBS','libs','Laser-induced breakdown','laser induced breakdown','LA-SIBS','filament-induced breakdown spectroscopy'], 
        '6':['terahertz', 'THz'], 
        '7':['fluoresc','fluorescence','fluorometer'],
        '8':['remote sensing', 'uav', 'aerial','satellite','airborne','AVIRIS','LiDAR','radar','chime','Sentinel','drones','SATllite','Landsat','Gaofen','Earth Observing','aircraft','air-borne','GF-5','WorldView','drone','AISA EAGLE','AISA','remote-sensing','GIIRS','ALOS-2 PALSAR','Earth observation','GOCI-II','MODIS','SPOT 4','GOCI','uas','GF-1','GEE','GF-6','SPOT-7','RapidEye','H-8'],
        '9':['else']
        }
        print('exact_match loaded.')
        return self.EXACT_MATCH

    def more_exact_match(self):
        self.EXACT_MATCH={
        '1':['uv' , 'ultravio'], 
        '2':['vis','visible'], 
        '3.1':['nir','near-infrared','near infrared','near infra-red','SWIR'], 
        '3.2':['mir','mid-infrared','mid-ir','mid infrared','mid-DRIFTS','midinfrared'], 
        '3.3':['ftir','Fourier transform infrared','ft-ir' ,'Fourier-transform infrared'], 
        '3.4':['long-wave infrared','long wave infrared','lwir','lir'],
        '4':['raman','SERS'], 
        '5':['LIBS','libs','Laser-induced breakdown','laser induced breakdown','LA-SIBS','filament-induced breakdown spectroscopy'], 
        '6':['terahertz', 'THz'], 
        '7':['fluoresc','fluorescence','fluorometer'],
        '8':['remote sensing', 'uav', 'aerial','satellite','airborne','AVIRIS','LiDAR','radar','chime','Sentinel','drones','SATllite','Landsat','Gaofen','Earth Observing','aircraft','air-borne','GF-5','WorldView','drone','AISA EAGLE','AISA','remote-sensing','GIIRS','ALOS-2 PALSAR','Earth observation','GOCI-II','MODIS','SPOT 4','GOCI','uas','GF-1','GEE','GF-6','SPOT-7','RapidEye','H-8'],
        '9':['else']
        }
        print('exact_match loaded.')
        return self.EXACT_MATCH

    def sample_match(self,
                     sample:dict,
                     ref_dict:dict,
                     verbose=False):
        kind_key = sample['Spectral Kind'].split(' :')[0]
        kind_key = self.str_to_list(kind_key)[0]
        print(f'{kind_key}') if verbose else None
        Output = sample['Output']
        model_response = sample['model_response']
        differing_interpretations = [i for i in ref_dict[kind_key]]
        if verbose:
            print(f'Output : {Output}')
            print(f'model_response : {model_response}')
            print(f'differing_interpretations : {differing_interpretations}')
        result = self.match(model_response,differing_interpretations,verbose=verbose)
        return result

    def match(self,source_string, differing_interpretations,verbose=False):
        is_found = False
        Pair = []
        for i in differing_interpretations:
            contains_string = i
            if re.search(contains_string, source_string, re.IGNORECASE):
                is_found = True
                Pair.append([source_string,contains_string])
        if verbose:
            if is_found:         
                print('Found')
                print(Pair) 
            else:print('Not Found')
        return is_found

    def str_to_list(self,str):
        str = str.replace('[','').replace(']','').replace('\'','')
        str = str.split(',')
        return str

    def save_items(self,
                   score:Union[float,list,Dict],
                   method_name:str,
               ) -> Dict:
        record_score = {'{}'.format(method_name):score}
        return record_score

    def save_json(self,
                  result:dict,
                  path:str,
                  ) -> None:
        json.dump(result,open(path,'w'))
