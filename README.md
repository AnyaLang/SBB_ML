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

**Results of the training on the batch of 16 with different learning rates**

| Learning Rate | Epoch | Average Loss | Accuracy   | Precision | Recall    | F1 Score  | Notes                  |
|---------------|-------|--------------|------------|-----------|-----------|-----------|------------------------|
| 0.0001        | 1     | 0.15489      | 22.71%     | 18.81%    | 22.71%    | 12.83%    | -                      |
| 0.0001        | 2     | 0.10375      | 37.92%     | 46.40%    | 37.92%    | 32.24%    | -                      |
| 0.0001        | 3     | 0.09681      | 45.62%     | 52.56%    | 45.62%    | 42.65%    | -                      |
| 0.0001        | 4     | 0.07292      | 51.15%     | 50.92%    | 51.15%    | 50.50%    | -                      |
| 5e-05         | 1     | 0.10961      | 42.92%     | 39.80%    | 42.92%    | 39.47%    | -                      |
| 5e-05         | 2     | 0.08257      | 50.42%     | 53.07%    | 50.42%    | 48.79%    | -                      |
| 5e-05         | 3     | 0.06633      | 55.31%     | 57.52%    | 55.31%    | 54.84%    | -                      |
| 5e-05         | 4     | 0.05221      | 56.98%     | 57.05%    | 56.98%    | 56.76%    | Best overall performance |
| 3e-05         | 1     | 0.10324      | 44.58%     | 52.58%    | 44.58%    | 42.50%    | -                      |
| 3e-05         | 2     | 0.07793      | 48.96%     | 49.55%    | 48.96%    | 46.42%    | -                      |
| 3e-05         | 3     | 0.06602      | 55.62%     | 56.60%    | 55.62%    | 55.01%    | -                      |
| 3e-05         | 4     | 0.05735      | 55.94%     | 55.52%    | 55.94%    | 55.39%    | -                      |
| 2e-05         | 1     | 0.10434      | 44.90%     | 50.87%    | 44.90%    | 42.50%    | -                      |
| 2e-05         | 2     | 0.07980      | 49.58%     | 50.54%    | 49.58%    | 48.43%    | -                      |
| 2e-05         | 3     | 0.06963      | 53.44%     | 53.24%    | 53.44%    | 52.64%    | -                      |
| 2e-05         | 4     | 0.06303      | 52.71%     | 52.48%    | 52.71%    | 52.28%    | -                      |

Our training was interrupted, so we couldn't fully evaluate the results for a batch size of 32. However, from what we observed, a learning rate of 5e-05 yielded the highest performance in terms of accuracy, precision, recall, and F1 score. In the next stage, we will continue training our model using this learning rate and adjust the batch size to fully assess the results for batch 32.

We further trained the model on the batch size 32 over 4 epochs with the learning rate 5e-05. These are the results that we got:

| Epoch | Learning Rate | Average Loss     | Accuracy   | Precision | Recall   | F1 Score  |
|-------|---------------|------------------|------------|-----------|----------|-----------|
| 1/4   | 5e-5          | 0.053033780585974| 0.43125    | 0.5486596 | 0.43125  | 0.4053166 |
| 2/4   | 5e-5          | 0.040040116741632| 0.4791667  | 0.5172350 | 0.4791667| 0.4742365 |
| 3/4   | 5e-5          | 0.031953962224846| 0.5510417  | 0.5744112 | 0.5510417| 0.5477183 |
| 4/4   | 5e-5          | 0.025974183475288| 0.5739583  | 0.5810352 | 0.5739583| 0.5750711 |

![alt text](http://url/to/img.png)

