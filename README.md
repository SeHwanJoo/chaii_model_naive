# chaii_model_naive

[kaggle chaii - Hindi and Tamil Question Answering](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/overview)

[top 7th solution](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/287948)

### Competition overview
chaii is an abbreviation for Challenge in AI for India.
The goal of this competition is to predict the question answering of the Hindi and tamil languages.

### Model
I use 5 XLM-roberta-large, 5 Muril-large, 4 Rembert.

#### XLM-Roberta-large
This model's tokenizer contains 250,002 vocab. And perform well in squad 2 dataset.

paper: https://arxiv.org/pdf/1911.02116.pdf

huggingface: https://huggingface.co/deepset/xlm-roberta-large-squad2

#### Muril-large
This model design for Indian language. I will pretrain this model on squad 2 dataset.

paper: https://arxiv.org/abs/2103.10730

huggingface: https://huggingface.co/google/muril-large-cased

#### Rembert
This model's tokenizer contains 250,300 vocab. It designs for multi-language and perform well. I will pretrain this model on squad 2 dataset.

paper: https://arxiv.org/abs/2010.12821

huggingface: https://huggingface.co/google/rembert 


### Train
#### XLM-roberta-large
I take this model from [huggingface](https://huggingface.co/deepset/xlm-roberta-large-squad2).
This model performed well on competition dataset. Because model is already trained on Question Answering dataset(squad2).
And just finetune model on competition dataset.

#### Muril-large
I use this model from [official repository](https://huggingface.co/google/muril-large-cased).
This model performed not good on competition dataset. So I train this model on Question Answering dataset(squad2).
And finetune the model on competition dataset.

#### Rembert
This model have same train process of muril.

### ensemble
I use 5 XLM-roberta-large model with no fold out and different seed.
I use 5 Muril-large model with 5 fold and same seed.
I use 4 Rembert model with 4 fold and same seed.

This models are have difference tokenizer. So I can't ensemble base on output logits. And hard vote across all models.

Then each kind of model can ensemble base on output logits, becuase of same tokenizer. So I use each model's ensemble outputs.

As a result, my final submission is as follows.

Final 1: 5 XLM + 5 Muril + 4 Rembert

Final 2: 5 XLM + ensembled XLM + 5 Muril + ensembled Muril + Rembert

### did not work
- different seed ensemble on Muril or Rembert
- translate augmentation
- remove stop word
- synonym augmentation
- score base ensemble
- start output, end output hard vote ensemble
- 10 models ensemble

### Result
There are many shake up and down in this competition. But I can shake up from 13th place to 7th place. I think this is because I use many kind of models and consider a lot of ensemble method.

