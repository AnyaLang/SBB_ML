# SBB_ML

7️⃣ **Doc2Vec**

In our Doc2Vec model, each word in the corpus is represented as a unique, high-dimensional vector. These vectors are trained such that words appearing in similar contexts have vectors that are close to each other in vector space. This characteristic allows the model to capture semantic relationships between words based on their usage in the text. We decided to explore which words our model finds semantically similar. We decided to look at the word "jour"

*Similar Words to 'jour':*
- paris: 0.9852
- tard: 0.9822
- matin: 0.9821
- vacance: 0.9809
- après-midi: 0.9809
- coucher: 0.9807
- cinéma: 0.9796
- habiter: 0.9794
- sport: 0.9793
- voir: 0.9790

**1. Data Preparation** Prior to deploying the Doc2Vec model, essential preprocessing steps were executed to prepare French language texts, which are crucial for optimizing model performance:

*Tokenization and Cleaning:* Implemented a custom tokenizer using spaCy to:
- Lemmatize words to their base forms.
- Remove stopwords, punctuation, and placeholders like 'xx' or 'xxxx'.
- Convert all text to lowercase to ensure uniformity.
  
*Example Transformation:* 

`sample_sentence = "Nous sommes l'équipe SBB et nous faisons de notre mieux pour développer la meilleure machine d'apprentissage automatique pour la classification des phrases."`

`processed_text = spacy_tokenizer(sample_sentence)`

`processed_text`

Transformed the sentence to *"équipe sbb faire mieux développer meilleur machine apprentissage automatique classification phrase"*

**2. Feature Selection and Model Setup** Doc2Vec has several parameters that can be tuned:
- *vector_size:* Dimensionality of the feature vectors. Values assessed: 50, 100
- *window:* The maximum distance between the current and predicted word within a sentence. Values assessed: 2, 5, 8
- *min_count:* Ignores all words with total frequency lower than this. Values assessed: 1 (no words are ignored based on frequency), 2, 5
- *workers:* Use these many worker threads to train the model (faster training with multicore machines). Values assessed: 4
- *alpha:* The initial learning rate. Values assessed: 0.025
- *min_alpha:* Learning rate will linearly drop to min_alpha as training progresses. Values assessed: 0.00025
- *epochs:* Number of iterations over the corpus. Values assessed: 40, 60, 100
- Additional training algorithms, such as *setting dm (distributed memory model) or DBOW (distributed bag of words).* Values assessed: dm = 0; dm = 1 (model uses the context window to predict the next word (Distributed Memory model, DM)), additional dm_concat=1

**3. Model Classifier** 

For classification, we tested different configurations of logistic regression to determine the optimal setup for our predictive tasks:

- *Basic Logistic Regression*: Initially, we deployed a logistic regression with default parameters to establish a baseline for performance comparison.
- *Regularized Logistic Regression with L2 Penalty*:
Configuration: LogisticRegression(C=10, penalty='l2', solver='saga')
Purpose: We increased the regularization strength to C=10 and employed the l2 penalty.
- *Regularized Logistic Regression with L1 Penalty*:
Configuration: LogisticRegression(C=10, penalty='l1', solver='saga')
Purpose: To assess the impact of l1 regularization, which promotes sparsity in the model coefficients, potentially improving model interpretability and performance on sparse data sets.

We used "saga" as it showed slightly higher results than "lbfgs" in this scenario.

**4. Model Evaluation and Results** 

1) **Regularization and Solver:** Configurations using {'C': 10, 'penalty': 'l2'} slightly outperformed those with {'C': 1, 'penalty': 'l1'}, indicating that stronger regularization with L2 penalty (which also promotes feature sparsity) might be more effective for this dataset.
2) **Epochs:** Longer training periods (100 epochs) sometimes resulted in slightly improved accuracy but not consistently across all configurations.
3) **Frequency count** Increasing min_count did not lead to an increase in the accuracy
4) **Window** Larger window sizes led to higher accuracy from 2 to 8. This suggests that considering a broader context around each word helps the model to better understand the text and make more accurate predictions.
5) **Vector_size**: With the vector size increase from 50 to 100, the accuracy of the model substantially increases
6) **Training algorithms** DBOW (dm=0) performed better than DM (dm=1)

The best accuracy observed was 44%, achieved with a configuration of 100-dimensional vectors, an 8-word window, min count of 1, 100 epochs, and logistic regression with C=10 and L1 penalty. This configuration suggests that higher dimensional vectors and more extensive training (more epochs) with stronger regularization might help in capturing more complex patterns in the data effectively.

*Best model configuration*

| Parameter      | Value                 |
|----------------|-----------------------|
| Vector Size    | 100-dimensional       |
| Window         | 8-word                |
| Minimum Count  | 1                     |
| Epochs         | 100                   |
| Classifier     | Logistic Regression   |
| C              | 10                    |
| Penalty        | L1                    |

**Confusion Matrix:**

|       | C1 | C2 | C3 | C4 | C5 | C6 |
|-------|----|----|----|----|----|----|
| **C1**|111 | 30 | 20 |  1 |  1 |  3 |
| **C2**| 43 | 65 | 31 | 11 |  2 |  6 |
| **C3**| 25 | 34 | 53 | 27 | 12 | 15 |
| **C4**|  7 |  6 | 22 | 60 | 27 | 31 |
| **C5**|  2 |  4 |  9 | 34 | 60 | 43 |
| **C6**|  3 |  3 | 23 | 14 | 48 | 74 |

While initially including a vector size of 200 in the evaluation loop, runtime issues occurred. This configuration was computed separately, revealing a model accuracy of 40%.

The initial results suggest that the combination of Doc2Vec and Logistic Regression provides a baseline for understanding and classifying our text data. We further adding more features such as TF-IDF scores to improve the model's understanding of the text.

The integration of **TF-IDF** features with **Doc2Vec** embeddings has improved the logistic regression model for text classification:

**TF-IDF Configuration:** `TfidfVectorizer(ngram_range=(1, 2))`

Despite using the best parameters from our previous configuration for the model with a TF-IDF matrix, we encountered computational limitations using LogisticRegression(C=10, penalty='l2', solver='saga'). A default version of logistic regression was subsequently used.

- **Combined Features Testing Configuration:** (100, 8, 1, 40)
- **Combined Features Accuracy:** 44.89%

**Confusion Matrix:**

|       | C1 | C2 | C3 | C4 | C5 | C6 |
|-------|----|----|----|----|----|----|
| **C1**|119 | 34 | 12 |  0 |  0 |  1 |
| **C2**| 60 | 67 | 26 |  2 |  2 |  1 |
| **C3**| 34 | 45 | 50 | 19 | 10 |  8 |
| **C4**| 17 |  7 | 21 | 52 | 32 | 24 |
| **C5**| 10 |  3 | 10 | 21 | 63 | 45 |
| **C6**| 12 |  5 |  9 | 30 | 29 | 80 |

**Classification Report**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| 0              | 0.45      | 0.68   | 0.54     | 166     |
| 1              | 0.39      | 0.39   | 0.39     | 158     |
| 2              | 0.38      | 0.30   | 0.33     | 166     |
| 3              | 0.45      | 0.38   | 0.41     | 153     |
| 4              | 0.48      | 0.40   | 0.44     | 152     |
| 5              | 0.51      | 0.52   | 0.52     | 165     |
| **Accuracy**   |           |        | **0.45**  | 960     |
| **Macro Avg**  | 0.44      | 0.44   | 0.44     | 960     |
| **Weighted Avg** | 0.44    | 0.45   | 0.44     | 960     |

**Conclusion**: The best accuracy we achieved was 44%, using a configuration that included 100-dimensional vectors, an 8-word window, a minimum count of 1, 100 epochs, and logistic regression with a regularization strength of C=10 and an L1 penalty. Given the computational constraints encountered during the training, we recognize that there is potential to achieve even higher accuracy, especially in the combination of the TF-IDF matrix. By using the default logistic regression and the same configuration for the Doc2Vec with TF-IDF matrix, we were able to achieve the 45% of accuracy of the model.

Our other findings indicated that the PV-DM model performed worse than the PV-DBOW model. This could be attributed to several factors, including the distinct configuration requirements for training each model or the possibility that our dataset is not large enough to capture meaningful contextual windows around each word. Additionally, the normalization processes applied to the text—such as lemmatization and the removal of stopwords and punctuation—might have eliminated crucial contextual elements. 

8️⃣ **BERT**

> As described on one of the [Hugging Face blogs](https://huggingface.co/blog/bert-101), BERT, short for Bidirectional Encoder Representations from Transformers, is a Machine Learning (ML) model for natural language processing. It was developed in 2018 by researchers at Google AI Language and serves as a **swiss army knife solution to 11+ of the most common language tasks**, such as sentiment analysis and named entity recognition.

And since we are in Switzerland 🇨🇭, deploying BERT seemed like a good idea to train the model for our task!

**1. Model architecture** BERT is a transformer-based machine learning model designed for natural language processing. In the following image, we can see the different sizes and architectures of the BERT model:

![Bert](https://huggingface.co/blog/assets/52_bert_101/BERT-size-and-architecture.png)

Here’s how many of the ML architecture components BERTbase and BERTlarge have:

| Model      | Transformer Layers | Hidden Size | Attention Heads | Parameters | Processing | Length of Training |
|------------|--------------------|-------------|-----------------|------------|------------|--------------------|
| BERTbase   | 12                 | 768         | 12              | 110M       | 4 TPUs     | 4 days             |
| BERTlarge  | 24                 | 1024        | 16              | 340M       | 16 TPUs    | 4 days             |


We used the base cased BERT model in our training due to computational limitations:

`tokenizer = BertTokenizer.from_pretrained('bert-base-cased')`
`model_bert = BertModel.from_pretrained('bert-base-cased')`

**2. Feature Selection and Model Setup** 

For text classification, we experimented with different configurations of the BERT model to determine the optimal setup for our predictive tasks:

- **Base BERT Model**:
  - **Configuration**: `AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=number_of_labels)`.
  We employed a base BERT model with default parameters to establish a baseline for performance comparison.
  
- **BERT with Increased Sequence Length**:
  - **Configuration**: impact of longer text sequences on model performance `max_length`. Values assessed: 128, 256
  
- **BERT with Additional Dropout**:
  - **Configuration**: Modified BERT configuration to include higher dropout rates.
 We explored the effect of increased regularization through dropout, aiming to reduce overfitting on the training data and improve generalization to new data.

We used different training strategies, focusing on adjustments in learning rates and epochs to optimize performance.

**3. Model evaluation** 

**Base BERT Model**:
The first training session was set up with a basic configuration with `AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=number_of_labels)` and `max_length = 128`. The parameters set for the training session were as follows:

```python
def tokenize_data(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
```

```python
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,  
    weight_decay=0.01
)
```

During our very first training with **simple configuration, we were able to already achieve 51% accuracy!**

*Results*
| Epoch | Training Loss | Validation Loss | Accuracy  | F1       | Precision | Recall   |
|-------|---------------|-----------------|-----------|----------|-----------|----------|
| 1     | No log        | 1.318586        | 40.2083%  | 32.3755% | 32.7760%  | 39.3142% |
| 2     | 1.310200      | 1.162180        | 51.0417%  | 49.3079% | 50.0007%  | 50.1226% |
| 3     | 1.310200      | 1.235953        | 51.4583%  | 50.3874% | 51.5714%  | 51.0867% |


The training loss remained stable, so it is worth using a more dynamic learning rate adjustment or a longer training period to see significant changes in loss reduction.

**BERT with Increased Sequence Length**:

For this model, we adjusted some of the parameters and also implemented a learning rate scheduler that gradually reduces the learning rate as the number of epochs increases. This can help the model fine-tune its weights more effectively towards the end of training.

1) Increased sequence length: `tokenizer = BertTokenizer.from_pretrained('bert-base-cased', model_max_length=256)`

2) Implemented the learning rate scheduler, which gradually reduces the learning rate as the number of epochs increases. In this case, no warm-up is added.

```python
scheduler = get_scheduler(
    "linear",
    optimizer=trainer.optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_epochs * len(train_dataset)
)
```
3) Other training arguments remained unchanged.

*Results*
| Epoch | Training Loss | Validation Loss | Accuracy  | F1       | Precision | Recall   |
|-------|---------------|-----------------|-----------|----------|-----------|----------|
| 1     | No log        | 1.221369        | 48.5417%  | 47.8542% | 51.3343%  | 48.5758% |
| 2     | 1.314400      | 1.149422        | 48.7500%  | 47.8679% | 48.1385%  | 48.5678% |
| 3     | 1.314400      | 1.172957        | 50.6250%  | 49.9922% | 50.8831%  | 50.3099% |


In this case, the accuracy became lower, but by using the increased sequence length and the learning rate scheduler we see a substantial decrease in the loss validation compared to our **Base BERT Model**. This can suggest that our model is generalizing better and not overfitting to the training data.

In the next step, we **increased the number of epochs to 5 epochs to monitor if there were any further improvements in the model accuracy.**

*Results*
| Epoch | Training Loss | Validation Loss | Accuracy  | F1       | Precision | Recall   |
|-------|---------------|-----------------|-----------|----------|-----------|----------|
| 1     | No log        | 1.235162        | 48.7500%  | 47.1810% | 52.0505%  | 48.5894% |
| 2     | 1.314300      | 1.147047        | 50.6250%  | 50.0270% | 50.5915%  | 51.0392% |
| 3     | 1.314300      | 1.153481        | 50.8333%  | 49.8434% | 51.3884%  | 50.9559% |
| 4     | 0.874100      | 1.205762        | 51.0417%  | 50.7684% | 51.5135%  | 50.7474% |
| 5     | 0.874100      | 1.277318        | 50.2083%  | 49.4280% | 50.3706%  | 49.5938% |



The training loss decreases significantly after the third epoch, indicating that the model continues to learn and improve its understanding of the training data as more epochs are processed. The validation loss does not show a clear decreasing trend; it increases slightly in the later epochs. This could suggest the beginning of overfitting. The accuracy is the highest in the 4th epoch.


**BERT with Additional Dropout**:
In addition to the previous configuration, the further modifications were made:

1) To prevent overfitting or underfitting, we included the dropout rate parameter

`config = BertConfig.from_pretrained('bert-base-cased', num_labels=len(label_dict), hidden_dropout_prob=0.2)`

2) Included the warmup stage to the scheduler.

2) Adjusted training arguments:
   
```python
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=1e-5,  # Adjusted learning rate
    per_device_train_batch_size=32,  # Adjusted batch size
    per_device_eval_batch_size=128,  # Adjusted batch size
    num_train_epochs=5,  # Increased number of epochs
    weight_decay=0.01
)
```

1) To prevent overfitting or underfitting, we included the dropout rate parameter

`config = BertConfig.from_pretrained('bert-base-cased', num_labels=len(label_dict), hidden_dropout_prob=0.2)`



Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	No log	1.452570	0.412500	0.369941	0.407669	0.407391
2	No log	1.316133	0.433333	0.407037	0.412354	0.423493
3	No log	1.280372	0.462500	0.446163	0.452032	0.451226
4	1.411800	1.298560	0.437500	0.416285	0.428490	0.429181
5	1.411800	1.354120	0.422917	0.401296	0.432286	0.415019


with all different things but same configuration:

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01
)

Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	No log	1.251293	0.479167	0.459857	0.494765	0.475285
2	1.360500	1.186942	0.493750	0.482764	0.492060	0.491113
3	1.360500	1.234582	0.481250	0.467804	0.491556	0.478042


**4. Results**

1) **Training Configuration**: Models with a higher sequence length (`max_length=256`) slightly outperformed those with the default length, indicating that capturing more context from the input text can be beneficial.
2) **Dropout and Regularization**: Introducing higher dropout rates improved model robustness, reducing overfitting as observed in validation loss improvements.
3) **Epochs**: Longer training periods (up to 5 epochs) generally resulted in improved accuracy, suggesting that the model benefits from more extensive training iterations to better learn the complexities of the dataset.
4) **Learning Rate**: Adjusting the learning rate to be lower showed better convergence over epochs without overshooting minima.
5) **Tokenization Effects**: Expanding the tokenizer's vocabulary and enabling more aggressive token cleaning techniques slightly increased model performance, hinting at the importance of input quality and preprocessing.
6) **Comparison with Traditional Classifiers**: BERT outperformed traditional machine learning classifiers (such as logistic regression) across most metrics, especially in handling nuanced language features and complex sentence structures.
7) **Training Strategy**: Employing dynamic masking and varied sentence lengths during training helped the model generalize better to unseen text, avoiding common pitfalls of fixed masking strategies.


While the capabilities of this model are extensive, we chose the FlauBERT model, which is more targeted towards our task, and therefore did not perform further hyperparameter tuning for BERT.

9️⃣ **FlauBERT**
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

### **FlauBERT Model over different learning rates:**

Important to note that our first model is trained using the **AdamW optimizer**, which is a variant of the traditional Adam optimizer. AdamW incorporates a regularization technique known as [weight decay](https://github.com/tml-epfl/why-weight-decay), which is used in training neural networks to prevent overfitting. It functions by incorporating a term into the loss function that penalizes large weights.

Besides, we also employ a **Linear Learning Rate Scheduler** to manage the learning rate throughout the training process. This scheduler starts with a relatively high learning rate and gradually decreases it to near zero by the end of the training. This approach ensures that we begin training with aggressive learning steps and fine-tune the model parameters more delicately as training progresses.

Although this training setup does not include a warm-up phase where the learning rate would gradually ramp up before decreasing, the scheduler is configured to reduce the learning rate slightly with each training step. This gradual reduction helps in stabilizing the training as it advances.

We would like to first observe the results with this setup and may adjust these parameters further based on the outcomes of the initial training phase.

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

The main difference between the performance of the training on the batch 16 and 32 with the same learning rate 5e-5 is the average loss. From the graph, it's clear that the average loss for the batch size of 32 is significantly lower than that for the batch size of 16 at every epoch.

![16 and 32 batch_loss.png](https://github.com/AnyaLang/SBB_ML/blob/125ad85ffe0c16a54ce6138367cc1b7af2dc0b6e/16%20and%2032%20batch_loss.png)

### **Model with 5e-5 learning rate, batch 32, 6 epochs**

We decided to train the model over a larger number of epochs, in this case 6 with the batch 32 and the same learning rate as before.

**Results with the batch 32, learning rate 5e-5 over larger number of epochs**

| Epoch | Learning Rate | Average Loss   | Accuracy   | Precision   | Recall   | F1 Score   |
|-------|---------------|----------------|------------|-------------|----------|------------|
| 1/6   | 5e-5          | 0.05631464881  | 0.4041667  | 0.4795025   | 0.4041667| 0.3879119  |
| 2/6   | 5e-5          | 0.04334156585  | 0.5333333  | 0.5451042   | 0.5333333| 0.5311008  |
| 3/6   | 5e-5          | 0.03416992826  | 0.546875   | 0.5649137   | 0.546875 | 0.5387779  |
| 4/6   | 5e-5          | 0.02753307774  | 0.6010417  | 0.6019167   | 0.6010417| 0.5942725  |
| 5/6   | 5e-5          | 0.02137682571  | 0.590625   | 0.5990724   | 0.590625 | 0.5874222  |
| 6/6   | 5e-5          | 0.01756872524  | 0.596875   | 0.6022174   | 0.596875 | 0.5978323  |

We then submitted two models on Kaggle, from the epoch 4 and 6. While the epoch 4 had higher accuracy, it provided the result of 0.573 on Kaggle. **For the model on the epoch 6th, the F1 score was higher, leading to the result of Kaggle of 0.601.** This demonstrates that relying solely on accuracy might not give a comprehensive assessment of a model's performance. Therefore, it is crucial to consider multiple metrics.

We also experimented and changed the number of epochs to 4, 6 and 8. However, 6 epochs resulted in the highest accuracy of the model and F1 value.

### **Model with a different learning rate adjustement**

Modifications for Warm-Up Phase and Learning Rate Adjustment

**Increased the Initial Learning Rate:** We start with a higher initial learning rate - 1e-4.

**Added Warm-Up Steps:** Introduce a warm-up phase where the learning rate will linearly increase to this higher initial rate over a number of steps. A common strategy is to set the warm-up steps to 10% of the total training steps.

`# Initialize the optimizer with a higher initial learning rate`
`optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)`

`# Scheduler with warm-up phase`
`scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)`

**Results for 8 epochs with the adjusted learning rate**

| Epoch | Learning Rate | Average Loss      | Accuracy   | Precision | Recall   | F1 Score  |
|-------|---------------|-------------------|------------|-----------|----------|-----------|
| 1/8   | 0.00009722    | 0.059470302890986 | 0.365625   | 0.3330621 | 0.365625 | 0.3027832 |
| 2/8   | 0.00008333    | 0.053232380685707 | 0.4791667  | 0.4275987 | 0.4791667| 0.4342403 |
| 3/8   | 0.00006944    | 0.038865594674523 | 0.5541667  | 0.5699164 | 0.5541667| 0.5361277 |
| 4/8   | 0.00005556    | 0.034336493226389 | 0.565625   | 0.5869896 | 0.565625 | 0.5635115 |
| 5/8   | 0.00004167    | 0.022284670624261 | 0.6010417  | 0.5997075 | 0.6010417| 0.5990875 |
| 6/8   | 0.00002778    | 0.015347646844263 | 0.603125   | 0.6056449 | 0.603125 | 0.5911651 |
| 7/8   | 0.00001389    | 0.009685143810930 | 0.6020833  | 0.6107261 | 0.6020833| 0.6032911 |
| 8/8   | 0.00000000    | 0.005019144429146 | 0.6166667  | 0.6212918 | 0.6166667| 0.6177674 |

![learning_rate.png](https://github.com/AnyaLang/SBB_ML/blob/b509447374760d91759c3c62027701d928a15ce2/Model%20with%20a%20different%20learning%20rate%20adjustement.png)

While the model with the adjusted learning rate demonstrated a higher accuracy score and performed better than the models before over other metrics, the submission on Kaggle provided a lower acore. We also adjusted the number of epochs to 15 and lower, however, the results were worse.

### **Model 3e-05 with large number of epochs and batch size 16**

We decided, to explore a bit more the training of the models over the lower batch size and different learning rates than before.

Each training session was conducted with a distinct learning rate, ranging from 1e-06 to 2e-05. The goal was to find an optimal rate that balances fast learning without overshooting the minimum of the loss function. For each learning rate, the model was trained over two epochs. This limited exposure was designed to quickly assess the impact of each learning rate without extensive computational cost.

| Learning Rate | Epoch | Average Loss    | Validation Accuracy |
|---------------|-------|-----------------|---------------------|
| 1e-05         | 1/2   | 1.776790196200212 | 0.3614583333333333  |
| 1e-05         | 2/2   | 1.4493117819229762| 0.4354166666666667  |
| 2e-05         | 1/2   | 1.8514792347947757| 0.43020833333333336 |
| 2e-05         | 2/2   | 1.3657454945147038| 0.5104166666666666  |
| 5e-06         | 1/2   | 2.1787539795041084| 0.3072916666666667  |
| 5e-06         | 2/2   | 1.7749672616521517| 0.346875            |
| 1e-06         | 1/2   | 2.78974984139204  | 0.19375             |
| 1e-06         | 2/2   | 2.3164451534549397| 0.20208333333333334 |

We achieved an accuracy of 51% within just two epochs using a learning rate of 2e-05. Encouraged by these results, we have decided to continue refining the model with this learning rate. To explore the model's capacity further, we plan to keep the batch size to 16 and adjust the learning rate to 3e-05, while extending the training period to 15 epochs.

**Model 3e-05 with large number of epochs and batch size 16**

| Epoch  | Learning Rate | Average Loss    | Validation Accuracy | Notes                                     |
|--------|---------------|-----------------|---------------------|-------------------------------------------|
| 1/15   | 3e-05         | 1.6861949928105 | 0.459375            | Saved as `best_model_lr3e-05_ep1_acc0.46.pt`  |
| 2/15   | 3e-05         | 1.2930432051420 | 0.5104167           | Saved as `best_model_lr3e-05_ep2_acc0.51.pt`  |
| 3/15   | 3e-05         | 1.1450499561926 | 0.5020833           |                                           |
| 4/15   | 3e-05         | 0.9551384929568 | 0.5479167           | Saved as `best_model_lr3e-05_ep4_acc0.55.pt`  |
| 5/15   | 3e-05         | 0.8847448159009 | 0.5552083           | Saved as `best_model_lr3e-05_ep5_acc0.56.pt`  |
| 6/15   | 3e-05         | 0.6622620061661 | 0.5541667           |                                           |
| 7/15   | 3e-05         | 0.5362344713571 | 0.5625              | Saved as `best_model_lr3e-05_ep7_acc0.56.pt`  |
| 8/15   | 3e-05         | 0.4089817595979 | 0.5875              | Saved as `best_model_lr3e-05_ep8_acc0.59.pt`  |
| 9/15   | 3e-05         | 0.3382450588358 | 0.5885417           | Saved as `best_model_lr3e-05_ep9_acc0.59.pt`  |
| 10/15  | 3e-05         | 0.2671806021050 | 0.5770833           |                                           |
| 11/15  | 3e-05         | 0.2069165084783 | 0.590625            | Saved as `best_model_lr3e-05_ep11_acc0.59.pt` |
| 12/15  | 3e-05         | 0.1845976701433 | 0.5927083           | Saved as `best_model_lr3e-05_ep12_acc0.59.pt` |
| 13/15  | 3e-05         | 0.1560351345979 | 0.5958333           | Saved as `best_model_lr3e-05_ep13_acc0.60.pt` |
| 14/15  | 3e-05         | 0.1160823275044 | 0.584375            |                                           |
| 15/15  | 3e-05         | 0.1117068582588 | 0.5885417           |                                           |

With this setting, we were able to achieve an accuracy of 0.590 on Kaggle. In the subsequent training session, we achieved an accuracy on Kaggle of 0.593. While we strive to make our code reproducible, some aspects of the model are outside our control and are influenced by a degree of randomness.

`best_model_path = 'best_model_lr3e-05_ep13_acc0.60.pt'  #the second time we run the code, our best model was in epoch 7`

`model.load_state_dict(torch.load(best_model_path, map_location=device))`

`model.to(device)`

`model.train()`

**We saved this best model and continued the training with a lower learning rate of 2e-05.**

For instance, in the example below we demonstrate the results per epoch after the training on 15 epochs with the learning rate 3e-05 and then continuing the training on the lower learning rate for 9 epochs.

| Epoch | Learning Rate | Average Loss   | Validation Accuracy |
|-------|---------------|----------------|---------------------|
| 1/9   | 2e-05         | 0.533422770320 | 0.5625              |
| 2/9   | 2e-05         | 0.269488133614 | 0.5614583333333333  |
| 3/9   | 2e-05         | 0.143803662904 | 0.5666666666666667  |
| 4/9   | 2e-05         | 0.137765339206 | 0.5739583333333333  |
| 5/9   | 2e-05         | 0.076707698969 | 0.56875             |
| 6/9   | 2e-05         | 0.065273187822 | **0.5979166666666667**  |
| 7/9   | 2e-05         | 0.045589126971 | 0.596875            |
| 8/9   | 2e-05         | 0.199297960250 | 0.5739583333333333  |
| 9/9   | 2e-05         | 0.160073978136 | 0.584375            |

The best model from this training had the result on Kaggle of 0.600.

**Our predictions made with the model 0.600 for Kaggle**

| Difficulty Level | Count |
|------------------|-------|
| A1               | 176   |
| A2               | 182   |
| B1               | 244   |
| B2               | 200   |
| C1               | 242   |
| C2               | 156   |

![predictions.png](https://github.com/AnyaLang/SBB_ML/blob/a61cf5434af67b37f57bf0cd083882fffb8aaa4a/all_predictions.png)

**Approach we took for the best model on Kaggle**

**After achieving initial results from 15 epochs with the learning rate 3e-05, we changed the learning rate to 1e-05 and continued the training for 3 more epochs. We saw the improvement of the model, so decided to proceed the training with slightly higher learning rate.**

We extended the training by **an additional 6 epochs with learning rate 2e-05, which further refined our model.** 

| Epoch | Learning Rate | Average Loss        | Validation Accuracy | Notes                                          |
|-------|---------------|---------------------|---------------------|------------------------------------------------|
| 1/6   | 2e-05         | 0.06369797926811316 | 0.5697916666666667  | Saved as `best_model_lr2e-05_ep1_acc0.57.pt`    |
| 2/6   | 2e-05         | 0.0697462880416424  | 0.5864583333333333  | Saved as `best_model_lr2e-05_ep2_acc0.59.pt`    |
| 3/6   | 2e-05         | 0.08821526710380567 | 0.5697916666666667  |                                                |
| 4/6   | 2e-05         | 0.03653059935331839 | 0.584375            |                                                |
| 5/6   | 2e-05         | 0.03376048295150819 | 0.5864583333333333  |                                                |
| 6/6   | 2e-05         | 0.02625617888628161 | 0.5916666666666667  | Saved as `best_model_lr2e-05_ep6_acc0.59.pt`    |

Observing continuous improvement, we decided that maintaining the learning rate of 2e-05 was optimal and proceeded to extend the training for 3 more epochs, however, given one issue in the code, the training extended to additional **9 epochs**. Throughout this extended training period, we noticed that while the **average loss consistently decreased, the accuracy improvements on our model plateaued, showing only marginal gains**.

| Epoch | Learning Rate | Average Loss         | Validation Accuracy   |
|-------|---------------|----------------------|-----------------------|
| 1/9   | 2e-05         | 0.049511629976404944 | 0.5791666666666667    |
| 2/9   | 2e-05         | 0.17178194310969655  | 0.5864583333333333    |
| 3/9   | 2e-05         | 0.03391529844190397  | 0.5927083333333333    |
| 4/9   | 2e-05         | 0.01702627820344181  | 0.5989583333333334    |
| 5/9   | 2e-05         | 0.049664503030241273 | 0.6020833333333333    |
| 6/9   | 2e-05         | 0.028027213982947313 | 0.59375               |
| 7/9   | 2e-05         | 0.01858836026416005  | 0.5947916666666667    |
| 8/9   | 2e-05         | 0.015126325636394237 | 0.59375               |
| 9/9   | 2e-05         | 0.03083539136728177  | 0.5927083333333333    |


The decrease in average loss indicates that our model was becoming better at fitting the training data, effectively minimizing the error between the predicted values and the actual values. This can be attributed to the model's increasing proficiency in identifying patterns and reducing prediction errors for the specific scenarios presented during training. However, the minimal gains in accuracy suggest that these improvements in loss did not translate to a broader generalization capability on unseen data. 

While our best model achieved an accuracy of 0.610 on Kaggle by following this approach, we did not fully rerun the model to generate the evaluation metrics of the final results due to computational limitations and financial constraints, as our team had to purchase computational units via Google Colab Pro. However, following the approach of initially training the model for 15 epochs with a learning rate of 3e-05, and then continuing the training with a starting with lower learning rate and adjusting it given the model results, should produce similar results.

| Description                     | Details                                        |
|---------------------------------|------------------------------------------------|
| **Initial Training**            | Learning rate of 1e-05                         |
| **Extended Fine-Tuning**        | Increased learning rate to 2e-05               |
| **Total Epochs**                | 15 (started with 6 and then 9)                 |
| **Epoch of Notable Outcome**    | 5 (from training with additional epochs)       |
| **Average Loss at Epoch 5**     | 0.049664503030241273                           |
| **Validation Accuracy at Epoch 5** | 0.6020833333333333                          |
| **Model Saved As**              | `best_model_lr2e-05_ep5_acc0.60.pt`            |
| **Final Loss**                  | 0.03083539136728177                            |
| **Final Accuracy**              | 0.6020833333333333                             |

**Our predictions made with the model 0.610 for Kaggle**

| Difficulty Level | Count |
|------------------|-------|
| A1               | 180   |
| A2               | 242   |
| B1               | 180   |
| B2               | 242   |
| C1               | 173   |
| C2               | 183   |

![predictions_best.png](https://github.com/AnyaLang/SBB_ML/blob/4f95efcdcfd33810b7f1419ee02da33ddcb365f1/best_graph.png)

## **Making predictions on the YouTube video**

We wanted to use our model to make the predictions on the videos targeted to beginner French learners. We chose the video on YouTube and created the df with the sentences from the video.

The video selected to make predictions is: [What Do French People Actually Eat? | Easy French 189](https://www.youtube.com/watch?v=p65EBC9lW9k&list=PLnazreCxpqRmpb4lGvzvCGXIXZkL3Nc27&index=4)

Example of the sentences:
`sentences = [
    "Salut les amis, bienvenue dans ce nouvel épisode.",
    "Aujourd'hui je suis dans le 11e arrondissement et je vais demander aux gens de parler de leurs habitudes alimentaires.",
    "Alors c'est parti, qu'est-ce que vous avez mangé depuis ce matin?",
]`

Since we could not access the models from the previous training due to a runtime interruption, we trained another model with a batch size of 32, for 6 epochs, and a learning rate of 5e-5 to make the predictions.

Based on the results after training the model on the dataset we obtained from Kaggle, we believe our model can well predict the difficulty of the sentences.

![youtube_predictions.png](https://github.com/AnyaLang/SBB_ML/blob/d42d9ce24795eb7c2b271b978b85b8798750944d/predictions_YouTube.png)

The video was produced for beginner French learners. From the plot, we can see that 16 sentences fall into the A2 category and 4 into the A1 category. Additionally, some sentences are classified as more difficult, at the B1 level, by the model. This classification could pose challenges for learners but also encourage them to acquire new vocabulary and further develop their language skills."
           
    accuracy                           0.45       960
   macro avg       0.44      0.44      0.44       960
weighted avg       0.45      0.45      0.45       960

![bert_confusion](https://github.com/AnyaLang/SBB_ML/blob/d793e080ecaa64a25a35fe512a9ef92109dd285e/confusion%20matrix_BERT.png)
