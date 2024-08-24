import evaluate
import re
import json
import numpy as np
import collections
from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Optional, Tuple, Union

class Evaluation_indicators:
    def __init__(self):
        self.dataset = None
        self.BLEU = evaluate.load("bleu")
        self.ROUGE = evaluate.load("rouge")
        self.BERT_SCORE = evaluate.load("bertscore")
        self.METEOR = evaluate.load('meteor')

    def load_data(self, predict_data_path: str = ''):
        if len(predict_data_path) == 0:
            raise TypeError("The path of predict data can't be null. Please enter the correct predictive data path.")
        else:
            self.predict_data_path = predict_data_path
        print(f'Loading data: "{self.predict_data_path}"')
        if self.predict_data_path.endswith(".json") or self.predict_data_path.endswith(".jsonl"):
            try:
                self.dataset = load_dataset("json", data_files=self.predict_data_path)['train']
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

    def parse_model_response(self, response):
        try:
            cleaned_response = response.rstrip('</s>').replace("'", '"')
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return {}

    def run_evaluate(self, evaluate_set=None, choose_methods=None, output_types=None):
        if output_types is None:
            output_types = {
                'entity1': 'Entity1:Object',
                'entity2': 'Entity2:Spectrum',
                'question_type': 'Question_type'
            }
        if choose_methods is None:
            choose_methods = {
                'entity1': ['bleu', 'rouge1','rouge2','rougeL', 'meteor', 'bert_score_precision','bert_score_recall','bert_score_f1'],
                'entity2': ['accuracy','bleu', 'rouge1','rouge2','rougeL','meteor', 'bert_score_precision','bert_score_recall','bert_score_f1'],
                'question_type': ['accuracy','bleu', 'rouge1','rouge2','rougeL', 'meteor', 'bert_score_precision','bert_score_recall','bert_score_f1']
            }

        Record = {}
        if not evaluate_set:
            evaluate_set = self.dataset

        for output_type, methods in choose_methods.items():
            output_column = output_types[output_type]
            for method in methods:
                results = []
                evaluator = getattr(self, method)() if method != 'accuracy' else None  # Do not load evaluator for accuracy
                #evaluator = getattr(self, method)()
                for entry in evaluate_set:
                    model_response = self.parse_model_response(entry['model_response'])
                    prediction = model_response.get(output_column)
                    reference = entry['Output'].get(output_column)
                    if prediction is not None and reference is not None:
                        predictions = [prediction]
                        references = [reference]
                        try:
                            eval_method = getattr(self, f"run_{method}")
                            result = eval_method(evaluator, predictions, references) if method != 'accuracy' else self.run_accuracy(predictions, references)
                            if isinstance(result, dict) and 'score' in result:
                                results.append(result['score'])
                            elif isinstance(result, (float, int)):
                                results.append(result)
                        except Exception as e:
                            print(f"Error computing {method} score: {e}")
                    else:
                        print(f"Missing prediction or reference for {output_column}")
                Record[f'{output_type}_{method}'] = np.mean(results) if results else 0

        return Record


    def run_accuracy(self, predictions, references):
        clean_predictions = [p for p in predictions if p is not None]
        clean_references = [r for r in references if r is not None]

        if len(clean_predictions) != len(clean_references) or not clean_predictions:
            print("预测和参考数据长度不匹配或为空")
            return 0
        try:
            correct = 0
            for pred, ref in zip(predictions, references):
        # 将预测和参考转换为字符串并检查是否互为子字符串
                if str(pred) in str(ref) or str(ref) in str(pred):
                    correct += 1
            total = len(predictions)
            accuracy = correct / total if total > 0 else 0
            return accuracy
        except Exception as e:
            print(f"计算 accuracy 分数时出错: {e}")
            return 0


    def run_bleu(self, evaluator, predictions, references):
        if not predictions or not references:
            print("Predictions or references are missing.")
            return 0
        print("Predictions:", predictions)
        print("References:", references)

        if not all(isinstance(ref, list) for ref in references):
            references = [[r] for r in references]
        try:
            result = evaluator.compute(predictions=predictions, references=references)
            return result['bleu']
        except Exception as e:
            print(f"Error computing BLEU score: {e}")
            return 0

    def run_rouge1(self, evaluator, predictions, references):
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        clean_predictions = [p for p in predictions if p is not None]
        clean_references = [r for r in references if r is not None]

        if len(clean_predictions) != len(clean_references) or not clean_predictions:
            print("预测和参考数据长度不匹配或为空")
            return 0
        try:
            result = evaluator.compute(predictions=clean_predictions, references=clean_references, rouge_types=["rouge1", "rouge2", "rougeL"])
            rouge_scores['rouge1'].append(result['rouge1'])
            rouge_scores['rouge2'].append(result['rouge2'])
            rouge_scores['rougeL'].append(result['rougeL'])
            return result['rouge1']
        except Exception as e:
            print(f"计算 ROUGE 分数时出错: {e}")
            return 0
        
    def run_rouge2(self, evaluator, predictions, references):
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        clean_predictions = [p for p in predictions if p is not None]
        clean_references = [r for r in references if r is not None]
        if len(clean_predictions) != len(clean_references) or not clean_predictions:
            print("预测和参考数据长度不匹配或为空")
            return 0
        try:
            result = evaluator.compute(predictions=clean_predictions, references=clean_references, rouge_types=["rouge1", "rouge2", "rougeL"])
            rouge_scores['rouge1'].append(result['rouge1'])
            rouge_scores['rouge2'].append(result['rouge2'])
            rouge_scores['rougeL'].append(result['rougeL'])
            return result['rouge2']
        except Exception as e:
            print(f"计算 ROUGE 分数时出错: {e}")
            return 0

    def run_rougeL(self, evaluator, predictions, references):
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        clean_predictions = [p for p in predictions if p is not None]
        clean_references = [r for r in references if r is not None]
        if len(clean_predictions) != len(clean_references) or not clean_predictions:
            print("预测和参考数据长度不匹配或为空")
            return 0
        try:
            result = evaluator.compute(predictions=clean_predictions, references=clean_references, rouge_types=["rouge1", "rouge2", "rougeL"])
            rouge_scores['rouge1'].append(result['rouge1'])
            rouge_scores['rouge2'].append(result['rouge2'])
            rouge_scores['rougeL'].append(result['rougeL'])
            return result['rougeL']
        except Exception as e:
            print(f"计算 ROUGE 分数时出错: {e}")
            return 0


    def run_bert_score_precision(self, evaluator, predictions, references):
        clean_predictions = [p for p in predictions if p is not None]
        clean_references = [r for r in references if r is not None]
        if len(clean_predictions) != len(clean_references) or not clean_predictions:
            print("预测和参考数据长度不匹配或为空")
            return 0
        try:
            result = evaluator.compute(predictions=clean_predictions, references=clean_references, lang='en')
            return np.mean(result['precision'])
        except Exception as e:
            print(f"计算 Bert_score 分数时出错: {e}")
            return 0

    def run_bert_score_recall(self, evaluator, predictions, references):
        clean_predictions = [p for p in predictions if p is not None]
        clean_references = [r for r in references if r is not None]

        if len(clean_predictions) != len(clean_references) or not clean_predictions:
            print("预测和参考数据长度不匹配或为空")
            return 0
        try:
            result = evaluator.compute(predictions=clean_predictions, references=clean_references, lang='en')
            return np.mean(result['recall'])
        except Exception as e:
            print(f"计算 Bert_score 分数时出错: {e}")
            return 0

    def run_bert_score_f1(self, evaluator, predictions, references):
        clean_predictions = [p for p in predictions if p is not None]
        clean_references = [r for r in references if r is not None]

        if len(clean_predictions) != len(clean_references) or not clean_predictions:
            print("预测和参考数据长度不匹配或为空")
            return 0
        try:
            result = evaluator.compute(predictions=clean_predictions, references=clean_references, lang='en')
            return np.mean(result['f1'])
        except Exception as e:
            print(f"计算 Bert_score 分数时出错: {e}")
            return 0

    def run_meteor(self, evaluator, predictions, references):
        clean_predictions = [p for p in predictions if p is not None]
        clean_references = [r for r in references if r is not None]

        if len(clean_predictions) != len(clean_references) or not clean_predictions:
            print("预测和参考数据长度不匹配或为空")
            return 0
        try:
            result = evaluator.compute(predictions=clean_predictions, references=clean_references)
            return result['meteor']
        except Exception as e:
            print(f"计算 Meteor 分数时出错: {e}")
            return 0

    def bleu(self):
        print('bleu loaded')
        return self.BLEU
    
    def rouge1(self):
        print('rouge loaded')
        return self.ROUGE
    
    def rouge2(self):
        print('rouge loaded')
        return self.ROUGE

    def rougeL(self):
        print('rouge loaded')
        return self.ROUGE

    def bert_score_precision(self):
        print('bert_score loaded')
        return self.BERT_SCORE

    def bert_score_recall(self):
        print('bert_score loaded')
        return self.BERT_SCORE
    
    def bert_score_f1(self):
        print('bert_score loaded')
        return self.BERT_SCORE    

    def meteor(self):
        print('meteor loaded')
        return self.METEOR


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
