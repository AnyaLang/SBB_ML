# SBB_ML

## **Training the model with the FlauBERT model**

For our training, we are using the FlauBERT model from [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/flaubert), as described:

> The FlauBERT model was proposed in the paper FlauBERT: Unsupervised Language Model Pre-training for French by Hang Le et al. It’s a transformer model pretrained using a masked language modeling (MLM) objective (like BERT).

For the FlauBERT model, one can choose from several options:

| Model name                | Number of layers | Attention Heads | Embedding Dimension | Total Parameters |
|---------------------------|------------------|-----------------|---------------------|------------------|
| flaubert-small-cased      | 6                | 8               | 512                 | 54M              |
| flaubert-base-uncased     | 12               | 12              | 768                 | 137M             |
| flaubert-base-cased       | 12               | 12              | 768                 | 138M             |
| flaubert-large-cased      | 24               | 16              | 1024                | 373M             |

We will use the FlauBERT large cased model because it has greater depth, a more sophisticated attention mechanism, a larger embedding size, and a higher parameter count. Larger models such as the FlauBERT large cased typically outperform smaller ones across a variety of language understanding benchmarks, potentially offering higher accuracy.

Also, the [BERT authors recommend fine-tuning](https://github.com/google-research/bert) for four epochs with the following hyperparameter options:

- batch sizes: 8, 16, 32, 64, 128
- learning rates: 3e-4, 1e-4, 5e-5, 3e-5

Given the computational limitation, we will train our model only on 2 different different batch sizes.
