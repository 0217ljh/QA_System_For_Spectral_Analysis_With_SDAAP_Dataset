
# LLM-based Spectral Detection Codebase

This repository contains the code for the paper "A Quick, trustworthy spectral detection Q&A system based on the SDAAP Dataset and large language model". The project focuses on utilizing two LLM (LLM1 and LLM2) for spectral detection tasks, particularly in entity extraction and response generation. The code is divided into three main directories, each with a specific role in the overall pipeline.



## SDAAP Dataset

![SDAAP](.\Picture\SDAAP.png 'SDAAP')

## Framework

![Framework](.\Picture\Framework.png 'Basic Q&A Framework')

## Repository Structure

- **LLM1_code**: This directory contains the code for LLM1, responsible for extracting relevant entities from the input questions.
  
- **Generate Dataset**: This directory includes the code required to generate the necessary datasets for training and testing the models. The datasets include input questions and the corresponding entities that need to be extracted.

- **LLM2_code**: This directory holds the code for LLM2, which generates responses based on the extracted entities and the knowledge base. It integrates the output of LLM1 to produce contextually relevant answers.

## Requirements

- accelerate
- appdirs
- loralib
- black
- black[jupyter]
- datasets
- fire
- torch==2.1.2
- peft==0.5.0
- transformers==4.36.2
- scipy
- bitsandbytes
- evaluate
- evaluation
- sentencepiece
- gradio
- wandb
- scikit-learn
- ipywidgets
- nltk
- rouge_score
- bert_score

You can install the necessary dependencies with:
```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/LLM-spectral-detection.git
    cd LLM-spectral-detection
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Generate Dataset

Navigate to the `Generate Dataset` folder and run the following notebooks in order:

- `Gen_Question.ipynb`
- `Gen_answer_type1.ipynb`
- `Gen_answer_type2.ipynb`
- `Gen_answer_type3.ipynb`

These notebooks will generate the necessary datasets for training and testing.

### 2. LLM1: Entity Extraction

Navigate to the `LLM1_code` folder and:

1. Run `main.ipynb` to train the model using the training set.
    The trained models will be saved in the Train folder under a directory named by the corresponding date.
2. Run `Gen_and_evl.ipynb` for inference and evaluation on the test set.
    The inference results and evaluation results will be saved in the Generate/output folder.
### 3. LLM2: Response Generation

Navigate to the `LLM2_code` folder and:

1. Run `main.ipynb` to train the model using the training set.
    The trained models will be saved in the Train folder under a directory named by the corresponding date.
2. Run `Gen_and_evl.ipynb` for inference and evaluation on the test set.
    The inference results and evaluation results will be saved in the Generate/output folder.

## Citation

If you use this code in your research, please cite our paper:

```
@article{your_paper,
  title={LLM Fine-tuning in Spectral Detection},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024},
  publisher={Publisher}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
