# Topic Modeling Pipeline

This folder contains a topic modeling pipeline that uses BERTopic to decompose a corpus into topics. The pipeline supports two datasets: a dummy dataset and the 20newsgroups dataset.

## Overview

The pipeline uses various embedding models to generate document embeddings, which are then clustered using BERTopic. The following models are used:

- **all-MiniLM-L6-v2** (22M parameters) - A lightweight sentence transformer model. [Citation: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, Reimers & Gurevych, 2019]
- **distilbert-base-uncased** (66M parameters) - A distilled version of BERT. [Citation: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter, Sanh et al., 2019]
- **bert-base-uncased** (110M parameters) - The original BERT model. [Citation: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al., 2019]
- **roberta-base** (125M parameters) - An optimized version of BERT. [Citation: RoBERTa: A Robustly Optimized BERT Pretraining Approach, Liu et al., 2019]
- **microsoft/MiniLM-L12-H384-uncased** (33M parameters) - A very lightweight model. [Citation: MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-trained Transformers, Wang et al., 2020]
- **meta-llama/Llama-2-7b-hf** (7B parameters) - Llama 2 7B model. [Citation: Llama 2: Open Foundation Language Models, Meta AI, 2023]
- **meta-llama/Llama-2-13b-hf** (13B parameters) - Llama 2 13B model. [Citation: Llama 2: Open Foundation Language Models, Meta AI, 2023]

## Metrics

The pipeline evaluates the topic models using the following metrics:

- **Topic Coherence**: Measures the semantic coherence of the topics using gensim's CoherenceModel with the 'c_v' coherence measure.
- **Topic Diversity**: Calculates the average pairwise cosine similarity between topics, converted to a diversity score (1 - similarity).
- **Number of Topics**: The number of unique topics identified by the model.
- **Running Time**: The time taken (in seconds) for the full pipeline per encoder.

## Usage

To run the pipeline, use the following command:

```bash
python experiments/topic_model_pipeline.py --dataset [dummy|20newsgroups] [--subset train|test|all] [--remove-headers] [--categories category1 category2 ...]
```

### Example

To run the pipeline on the 20newsgroups dataset with specific categories:

```bash
python experiments/topic_model_pipeline.py --dataset 20newsgroups --categories alt.atheism comp.graphics --remove-headers
```

## Dependencies

Ensure all dependencies are installed using:

```bash
pip install -r experiments/requirements.txt
```

## Output

The results are saved to `results.csv`, which includes the following columns:

- `encoder`: The name of the encoder used.
- `type`: The type of encoder (sentence-transformer or huggingface).
- `params`: Number of model parameters.
- `coherence`: Topic coherence score.
- `diversity`: Topic diversity score.
- `num_topics`: Number of unique topics.
- `running_time_sec`: Time taken for the pipeline in seconds. 

## Install packages using uv 

uv is "An extremely fast Python package and project manager, written in Rust". You can download it from [here](https://github.com/astral-sh/uv).

To run the code using uv (instructions for Windows but should be similar on other OS) you can:

1- Install uv followign the instructions here: https://github.com/astral-sh/uv

2- Clone this repository and go to the experiments folder

```bash
git clone https://github.com/willyrv/ml2nlp.git
```

```bash
cd ml2nlp\experiments
```

3- Create a virtual environment using uv

```bash
uv venv --python 3.11
```

4- Activate the virtual environment

```bash
.venv/bin/activate
```

5- Install the required packages

```bash
uv pip install -r requirements.txt
```

6- Run the python script

```bash
python .\topic_model_pipeline.py
```


