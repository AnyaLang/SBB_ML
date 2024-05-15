# Get to know French4U: Revolutionary Tech to Master French
Welcome to SBB LogoRank, a startup adventure initiated by Anna Dovha and Ana Llorens, two university friends who aim to transform how individuals engage with and master French. 

🇫🇷 **The Challenge**: The journey to fluency in French involves engaging with a wide range of texts that gradually expand one's vocabulary, comprehension, and comfort with the language. However, without guidance, learners can easily find themselves either bored with overly simple texts or frustrated by complex literature beyond their current understanding. This challenge poses a significant barrier to effective and enjoyable language learning.

💡 **Our Solution**: To address this challenge, we have developed an innovative model called "French4U" specifically designed for English speakers learning French. Our technology predicts the difficulty level of French-written texts, which empowers learners to choose reading materials that are perfectly aligned with their learning stage. This targeted approach ensures that every reading experience is both manageable and slightly challenging, promoting faster and more sustainable learning.

Next, we will walk you through the comprehensive steps, methodologies, and outcomes that shaped our journey to creating the final model, French4U. Are you ready to join us on this adventure?

## Methodologies and Results
### 1️⃣ Imports, Data Cleaning and Data Augmentation
The journey begins with the essential setup: importing libraries, data files, and other resources crucial for advancing to subsequent stages.

Among the imported data files, the 'training data' dataset is pivotal as it is used to train the models. However, before embarking on model development, we first go through a data cleaning and augmentation process:

**Data Cleaning**: While reviewing the sentences in the 'training data' dataset, we noticed that some lacked punctuation marks at the end. Recognizing the importance of punctuation for our upcoming data augmentation phase, we ensured that every sentence was properly punctuated:
- *"add_dot"*: Function designed to ensure that a given sentence ends with a punctuation mark. 

**Data Augmentation**: During this stage, we identified and added new features to the 'training data' dataset to further enhance sentence characterization. These features include:

- *"count_words"*: Number of words in a given sentence.
- *"avg_word_length"*: Average length of words in a given sentence.
- *"count_punctuation"*: Number of punctuation marks in a given sentence.
- *"stopword_proportion"*: Proportion of stop words (number of stop words/number of total words) in a given sentence.
- *"flesch_kincaid_readability"*: Readability score of a sentence using the Flesch-Kincaid Grade Level formula.
- *"get_pos_tags"* and *"analyze_pos"*: Analysis of the types of words comprising each sentence and assignment of part-of-speech (POS) tags to each word. For each tag, a new column is created in the dataset, and the corresponding value records the count of words with that specific POS tag present in each sentence.

These features will serve as additional input data alongside the following models. We will compare the results of models trained with these comprehensive features to those trained solely with the sentence as an input.

### 2️⃣ Logistic Regression
Our initial approach to predicting text difficulty levels involves employing logistic regression paired with TF-IDF vectorization. Like the subsequent methods we explore, this strategy has been implemented through two distinct evaluations to develop a model adept at accurately assessing the difficulty of texts written in French for English speakers. These evaluations are structured to assess and compare the efficacy of utilizing varied sets of input features for model training

*Sentence-Only Assessment:* This assessment involves training a model using only the sentence data as input (X). 

*Additional Features Assessment:* In this assessment, we broaden our methodology to incorporate additional linguistic and stylistic features into the analysis, in particular, those developed during the Data Augmentation phase previously described.

Here’s a step-by-step overview of the approach:

**1. Data Preparation and Label Encoding**
Firstly, we encode the 'difficulty' labels of our training data using LabelEncoder, transforming them into a machine-readable format. This step ensures that our model can effectively understand and process the target labels.

**2. Feature Selection and Model Setup**
We define our features (X) according to the type of assessment (sentence-only or with additional features) and our target (y) as the encoded difficulty levels. For text preparation, we employ the TF-IDF Vectorizer to convert text into a format emphasizing key distinguishing words for classification. Additionally, we use Standard Scaling to normalize the numeric values of the additional features. Both text and numeric data are integrated and processed using a ColumnTransformer, ensuring comprehensive and effective utilization for modeling.

**3. Model Pipeline Configuration**
In the sentence-only assessment, we implement a pipeline that includes TF-IDF vectorization followed by a logistic regression classifier. This configuration efficiently bridges the gap from data transformation to model training. Conversely, when the input includes additional features, we utilize the previously described ColumnTransformer for preprocessing. This is complemented by the LogisticRegression classifier, which serves as the final step in the pipeline, ensuring a cohesive and effective approach to handling mixed data types.

**4. Hyperparameter Tuning**
To optimize our model, we employ GridSearchCV with a range of parameters for TF-IDF and logistic regression:

- *"preprocessor__tfidf__ngram_range"*: This parameter determines the range of n-grams to be used for TF-IDF vectorization. An n-gram is a contiguous sequence of n items from a given sample of text or speech. For example, in text, an n-gram might be a sequence of words or letters. Values assessed: (1,1) and (1,2).
- *"preprocessor__tfidf__use_idf"*: This boolean parameter determines whether to use inverse document frequency (IDF) component in the TF-IDF vectorization. Values assessed: True and False.
- *"classifier__C"*: This is the inverse of regularization strength in logistic regression. Regularization is applied to avoid overfitting by penalizing larger coefficients. Values assessed: 0.1, 1 and 10.
- *"classifier__penalty"*: This parameter specifies the norm used in the penalization (regularization) of the logistic regression model. Values assessed: l2 and none.

> All the hyperparameter values mentioned in this document have been chosen with the intent of achieving optimal performance while also taking computational costs into account.

This extensive search help identifying an optimal combination of parameters for our model based on accuracy and by performing a 3-fold cross-validation.

**5. Training and Testing**
We split our dataset into training and testing sets to validate the effectiveness of our model. After training, we identify the best model parameters that lead to the highest cross-validation accuracy.

**6. Model Evaluation and Results**
Finally, we evaluate our best model on the test set to measure its performance. The classification reports obtained are the following ones: 

(1) *Classification Report: Only-sentence Assessment's Logistic Regression Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.58      | 0.60   | 0.59     | 153     |
| **1 (A2 Level)**| 0.49      | 0.45   | 0.47     | 156     |
| **2 (B1 Level)**| 0.39      | 0.44   | 0.41     | 153     |
| **3 (B2 Level)**| 0.55      | 0.35   | 0.43     | 173     |
| **4 (C1 Level)**| 0.47      | 0.39   | 0.43     | 166     |
| **5 (C2 Level)**| 0.46      | 0.69   | 0.55     | 159     |
| **accuracy**    |           |        | 0.48     | 960     |
| **macro avg**   | 0.49      | 0.49   | 0.48     | 960     |
| **weighted avg**| 0.49      | 0.48   | 0.48     | 960     |

Best parameters associated with this model:  {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs', 'tfidf__ngram_range': (1, 2), 'tfidf__use_idf': True}

(2) *Classification Report: Additional Features Assessment's Logistic Regression Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.58      | 0.71   | 0.64     | 153     |
| **1 (A2 Level)**| 0.47      | 0.44   | 0.46     | 156     |
| **2 (B1 Level)**| 0.49      | 0.50   | 0.50     | 153     |
| **3 (B2 Level)**| 0.48      | 0.51   | 0.49     | 173     |
| **4 (C1 Level)**| 0.49      | 0.37   | 0.42     | 166     |
| **5 (C2 Level)**| 0.55      | 0.54   | 0.55     | 159     |
| **accuracy**    |           |        | 0.51     | 960     |
| **macro avg**   | 0.51      | 0.51   | 0.51     | 960     |
| **weighted avg**| 0.51      | 0.51   | 0.51     | 960     |

Best parameters associated with this model:  {'classifier__C': 10, 'classifier__penalty': 'l2', 'preprocessor__tfidf__ngram_range': (1, 1), 'preprocessor__tfidf__use_idf': True}

**7. Conclusion**

The inclusion of additional features alongside the basic TF-IDF vectorization of the sentences appears to enhance the model's ability to more accurately and effectively classify the difficulty levels of French texts for English speakers. This is particularly evident in the improved accuracy (0.51 in the Additional Features Assessment VS 0.48 in the Only-Sentence Assessment) and macro averages, which suggest that the model benefits from a richer set of input data. These enhancements likely provide the model with a more nuanced understanding of the text, improving its performance, especially in correctly identifying instances at the extremes of the difficulty spectrum (A1 and C2 levels).

Regarding hyperparameters, both best models agreed that a regularization strength (classifier__C) of 10 was ideal, reflecting a preference for minimal regularization to allow greater flexibility in the model's decision boundaries. This level of regularization suggests that the models are capable of managing the balance between bias and variance without heavily penalizing the size of the coefficients. Additionally, the 'l2' penalty was chosen for both models, endorsing its effectiveness in controlling overfitting by squaring the coefficients, which proves beneficial regardless of the model configuration. Another shared setting was the use of inverse document frequency (IDF) in the TF-IDF vectorization, with both models performing optimally with IDF enabled. This setting underlines the importance of reducing the influence of frequently occurring terms, thereby enhancing the overall discriminatory power of the model's text processing.

However, the models diverged on the n-gram range used in TF-IDF vectorization. The Only Sentence model found that a range of (1, 2) — encompassing both unigrams and bigrams — was most effective, suggesting that the inclusion of bigrams provided significant contextual benefit when relying solely on textual data. In contrast, the Additional Features model opted for (1, 1), relying exclusively on unigrams. This indicates that the presence of additional numeric and categorical data diminished the need for capturing broader textual contexts through bigrams, as these contexts were adequately supplemented by the other features.

### 3️⃣ Random Forest Classifier
The procedure for implementing the Random Forest classifier closely mirrors that of the Logistic Regression, with a few key distinctions:

**Pipeline configuration:** In the pipeline, the Logistic Regression classifier is replaced by a RandomForestClassifier. Random Forest is an ensemble learning method based on decision trees, where multiple trees are generated during the training process, and their results are aggregated to provide the final output. This method is particularly noted for its robustness against overfitting and its ability to handle both linear and non-linear relationships.

**Parameter tuning specifics:** To enhance our model's performance, we once more utilize GridSearchCV, employing a 5-fold cross-validation strategy (as it offers a lower computational burden per fold compared to the previous approach) and some different parameters for TF-IDF and Random Forest Classifier:

- *"preprocessor__tfidf__ngram_range"*: This parameter is identical to the one specified for Logistic Regression. Values assessed: (1,1) and (1,2).
- *"preprocessor__tfidf__use_idf"*: This parameter is identical to the one specified for Logistic Regression. Values assessed: True and False.
- *"classifier__n_estimators"*: This parameter determines the number of decision trees in the random forest. Values assessed: 100, 200 and 300.
- *"classifier__max_depth"*: It sets the maximum depth of each decision tree in the forest. Values assessed: None, 10, 20 and 30.
- *"classifier__min_samples_split"*: This parameter specifies the minimum number of samples required to split an internal node. Values assessed: 2, 5 and 10.
- *"classifier__min_samples_leaf"*: It sets the minimum number of samples required to be at a leaf node. Values assessed: 1, 2 and 4.

**Results**

(1) *Classification Report: Only-sentence Assessment's Random Forest Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.41      | 0.78   | 0.54     | 153     |
| **1 (A2 Level)**| 0.35      | 0.29   | 0.32     | 156     |
| **2 (B1 Level)**| 0.40      | 0.36   | 0.38     | 153     |
| **3 (B2 Level)**| 0.39      | 0.34   | 0.36     | 173     |
| **4 (C1 Level)**| 0.33      | 0.27   | 0.30     | 166     |
| **5 (C2 Level)**| 0.49      | 0.36   | 0.41     | 159     |
| **accuracy**    |           |        | 0.40     | 960     |
| **macro avg**   | 0.40      | 0.40   | 0.39     | 960     |
| **weighted avg**| 0.40      | 0.40   | 0.38     | 960     |

Best parameters associated with this model: {'classifier__max_depth': None, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 300, 'tfidf__ngram_range': (1, 1), 'tfidf__use_idf': True}

(2) *Classification Report: Additional Features Assessment's Random Forest Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.55      | 0.77   | 0.64     | 153     |
| **1 (A2 Level)**| 0.41      | 0.40   | 0.41     | 156     |
| **2 (B1 Level)**| 0.39      | 0.41   | 0.40     | 153     |
| **3 (B2 Level)**| 0.40      | 0.32   | 0.36     | 173     |
| **4 (C1 Level)**| 0.38      | 0.36   | 0.37     | 166     |
| **5 (C2 Level)**| 0.51      | 0.43   | 0.47     | 159     |
| **accuracy**    |           |        | 0.45     | 960     |
| **macro avg**   | 0.44      | 0.45   | 0.44     | 960     |
| **weighted avg**| 0.44      | 0.45   | 0.44     | 960     |

Best parameters associated with this model:  {'classifier__max_depth': None, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 300, 'preprocessor__tfidf__ngram_range': (1, 1), 'preprocessor__tfidf__use_idf': True}

**Conclusion**

The Random Forest model trained on the "Only-sentence Assessment" achieved an accuracy of 40%, demonstrating moderate precision, recall, and F1-scores across proficiency levels. Notably, it showed better performance in distinguishing lower proficiency levels (A1, A2) but still indicated room for improvement in overall accuracy.

In contrast, the Random Forest model trained on the "Additional Features Assessment" outperformed its counterpart, achieving an accuracy of 45% with improved precision, recall, and F1-scores across all proficiency levels. Particularly, this model excelled in accurately classifying beginner proficiency levels (A1), showcasing the effectiveness of incorporating diverse features in proficiency assessment tasks.

Regarding the optimal hyperparameters, which turned out to be identical in both models, they represent a balanced trade-off between model complexity and generalization. With n_estimators set to 300, the models benefit from a larger ensemble size, enhancing stability and reducing overfitting. Setting max_depth to None allows trees to grow without restriction, capturing complex patterns. min_samples_split and min_samples_leaf values of 5 and 1, respectively, ensure robust splitting criteria and sufficient samples at leaf nodes for accurate predictions. Additionally, using unigrams (ngram_range=(1, 1)) with IDF weighting (use_idf=True) optimizes feature representation, emphasizing informative features while mitigating the impact of common terms. 

### 4️⃣ Decision Tree Classifier

The procedure for implementing the Decision Tree Classifier is also almost equal to that of the Logistic Regression, with a few key distinctions:

**Pipeline configuration:** In the pipeline, the Logistic Regression classifier is replaced by a DecisionTreeClassifier, which is a supervised learning algorithm used for classification tasks.

**Parameter tuning specifics:** To enhance our model's performance, we utilize GridSearchCV once more, employing a 5-fold cross-validation and the identical parameters as before for TF-IDF and the Decision Tree Classifier. However, we exclude "classifier__n_estimators", as it's not applicable in this scenario.

**Results**

(1) *Classification Report: Only-sentence Assessment's Decision Tree Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.34      | 0.62   | 0.44     | 153     |
| **1 (A2 Level)**| 0.26      | 0.23   | 0.24     | 156     |
| **2 (B1 Level)**| 0.18      | 0.14   | 0.16     | 153     |
| **3 (B2 Level)**| 0.30      | 0.40   | 0.34     | 173     |
| **4 (C1 Level)**| 0.32      | 0.17   | 0.22     | 166     |
| **5 (C2 Level)**| 0.41      | 0.19   | 0.24     | 159     |
| **accuracy**    |           |        | 0.29     | 960     |
| **macro avg**   | 0.28      | 0.29   | 0.28     | 960     |
| **weighted avg**| 0.29      | 0.29   | 0.28     | 960     |

Best parameters associated with this model: {'classifier__max_depth': 20, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'tfidf__ngram_range': (1, 1), 'tfidf__use_idf': True}

(2) *Classification Report: Additional Features Assessment's Decision Tree Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.57      | 0.66   | 0.61     | 153     |
| **1 (A2 Level)**| 0.42      | 0.50   | 0.46     | 156     |
| **2 (B1 Level)**| 0.28      | 0.28   | 0.28     | 153     |
| **3 (B2 Level)**| 0.36      | 0.28   | 0.31     | 173     |
| **4 (C1 Level)**| 0.35      | 0.27   | 0.30     | 166     |
| **5 (C2 Level)**| 0.40      | 0.45   | 0.42     | 159     |
| **accuracy**    |           |        | 0.40     | 960     |
| **macro avg**   | 0.40      | 0.41   | 0.40     | 960     |
| **weighted avg**| 0.39      | 0.40   | 0.40     | 960     |

Best parameters associated with this model: {'classifier__max_depth': 10, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 10, 'preprocessor__tfidf__ngram_range': (1, 1), 'preprocessor__tfidf__use_idf': False}

**Conclusion**

The Decision Tree model trained on the "Only-sentence Assessment" achieved low-moderate accuracy (29%) with varying precision, recall, and F1-scores across proficiency levels. In contrast, the "Additional Features Assessment" model significantly improved classification performance, achieving an accuracy of 40% and higher precision, recall, and F1-scores across all proficiency levels of French.

This improvement suggests again that incorporating additional features enhances the model's ability to accurately classify proficiency levels. Notably, the "Additional Features Assessment" model excelled in classifying beginner proficiency levels, demonstrating the importance of diverse features in proficiency assessment tasks.

Regarding the optimal hyperparameters, in the "Only-sentence Assessment" model, a deeper tree structure (max_depth=20) and less strict splitting criteria (min_samples_split=2) were chosen to potentially capture intricate patterns. Conversely, the "Additional Features Assessment" model opted for a shallower tree (max_depth=10) with stricter splitting criteria (min_samples_split=10) to prevent overfitting. Both models employed unigram TF-IDF representation, emphasizing single words' importance across documents. These hyperparameters were carefully tailored to each feature assessment's characteristics, highlighting the models' adaptability and the importance of parameter tuning for optimal classification performance.

### 5️⃣ KNN Classifier

The process of implementing the KNN classifier closely resembles that of Logistic Regression too, with a few key distinctions:

**Pipeline configuration:** In the pipeline, the Logistic Regression classifier is substituted with a KNN (K-Nearest Neighbors) classifier. KNN is a simple yet effective classification algorithm that classifies new data points based on the majority class of their neighboring data points.

**Parameter tuning specifics:** To optimize our model, we employ GridSearchCV again, with a 5-fold cross-validation and some different parameters for TF-IDF and KNN Classifier:

- *"tfidf__ngram_range"*: This parameter matches the one specified for Logistic Regression as *"preprocessor__tfidf__ngram_range"*. Values assessed: (1,1) and (1,2).
- *"tfidf__use_idf"*: This parameter matches the one specified for Logistic Regression as *"preprocessor__tfidf__use_idf"*. Values assessed: True and False.
- *"classifier__n_neighbors"*: This parameter determines the number of neighbors to consider for classification. KNN (K-Nearest Neighbors) assigns a class label to a new data point based on the majority class among its n_neighbors nearest neighbors. Values assessed: 3, 5, 6 (equal to the number of labels) and 7.
- *"classifier__weights"*: This parameter specifies the weight function used in prediction. It can take two values: 'uniform' (all neighbors are weighted equally in the prediction process) and 'distance'(closer neighbors are given more weight in the prediction, with weights inversely proportional to their distance from the query point).
- *"classifier__algorithm"*: This parameter specifies the algorithm used to compute the nearest neighbors. It can take four values: 'auto'(automatically selects the most appropriate algorithm based on the training data), 'ball_tree' (uses a Ball Tree data structure to perform nearest neighbor search efficiently), 'kd_tree' (uses a KD Tree data structure for efficient nearest neighbor search) and 'brute' (computes nearest neighbors by brute force, i.e., by comparing the distances to all training samples).

**Results**

(1) *Classification Report: Only-sentence Assessment's KNN Classifier Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.30      | 0.73   | 0.42     | 153     |
| **1 (A2 Level)**| 0.28      | 0.33   | 0.30     | 156     |
| **2 (B1 Level)**| 0.22      | 0.22   | 0.22     | 153     |
| **3 (B2 Level)**| 0.46      | 0.24   | 0.32     | 173     |
| **4 (C1 Level)**| 0.61      | 0.23   | 0.33     | 166     |
| **5 (C2 Level)**| 0.59      | 0.37   | 0.46     | 159     |
| **accuracy**    |           |        | 0.35     | 960     |
| **macro avg**   | 0.41      | 0.35   | 0.34     | 960     |
| **weighted avg**| 0.41      | 0.35   | 0.34     | 960     |

Best parameters associated with this model: {'classifier__algorithm': 'auto', 'classifier__n_neighbors': 3, 'classifier__weights': 'distance', 'tfidf__ngram_range': (1, 1), 'tfidf__use_idf': True}

(2) *Classification Report: Additional Features Assessment's KNN Classifier Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.50      | 0.71   | 0.59     | 153     |
| **1 (A2 Level)**| 0.38      | 0.38   | 0.38     | 156     |
| **2 (B1 Level)**| 0.33      | 0.26   | 0.29     | 153     |
| **3 (B2 Level)**| 0.34      | 0.29   | 0.31     | 173     |
| **4 (C1 Level)**| 0.32      | 0.29   | 0.30     | 166     |
| **5 (C2 Level)**| 0.36      | 0.39   | 0.38     | 159     |
| **accuracy**    |           |        | 0.38     | 960     |
| **macro avg**   | 0.37      | 0.39   | 0.37     | 960     |
| **weighted avg**| 0.37      | 0.38   | 0.37     | 960     |

Best parameters associated with this model: {'classifier__algorithm': 'auto', 'classifier__n_neighbors': 7, 'classifier__weights': 'distance', 'preprocessor__tfidf__ngram_range': (1, 1), 'preprocessor__tfidf__use_idf': False}

**Conclusion**

In conclusion, the KNN classifiers trained on two different feature assessments showed varying degrees of success in accurately classifying proficiency levels. Despite employing optimal hyperparameters and a thorough evaluation process, both models struggled with precision and recall, particularly for lower proficiency levels (A1, A2).

The "Additional Features Assessment" model exhibited slightly better performance compared to the "Only-sentence Assessment" model (0.35 vs 0.38), achieving higher accuracy and precision across most proficiency levels. This suggests that the inclusion of additional features contributed to marginal improvements in classification accuracy. However, both models still faced challenges in accurately distinguishing between different proficiency levels, indicating the need for further optimization and feature engineering.

Regarding optimal hypermeparameters, they turned out to be almost identical in both models except from *"classifier__n_neighbors"*. In the "Only-sentence Assessment" dataset, a smaller neighborhood size of 3 neighbors sufficed. This suggests that the feature space may be less complex, with neighboring samples providing sufficient contextual information for classification. Conversely, the "Additional Features Assessment" dataset incorporated additional and more diverse features, introducing greater variability and complexity to the feature space. As a result, a larger neighborhood size of 7 neighbors was necessary to capture the broader spectrum of feature interactions and relationships. This larger neighborhood enables the model to consider a wider array of samples during classification.

### 6️⃣ Neural Networks

In pursuit of higher accuracy levels, we have chosen to explore novel methodologies such as neural networks. This machine learning technique involves a series of steps, outlined as follows:

**1. Label Encoding:**
The 'difficulty' labels are encoded using a LabelEncoder to convert them into numerical format, suitable for model training.

**2. TF-IDF Vectorization:**
The sentences are transformed into TF-IDF (Term Frequency-Inverse Document Frequency) vectors using TfidfVectorizer. This step converts text data into numerical features, where each word's importance is weighted based on its frequency in the sentence and rarity across all sentences.

**3. Train-Test Split:**
The dataset is split into training and testing sets using train_test_split from sklearn.model_selection. This separation ensures that the model's performance can be evaluated on unseen data.

**4. Feature Scaling:**
The input features are scaled using MinMaxScaler (sentences) and StandardScaler (features) to ensure that all features are on a similar scale. This prevents certain features from dominating others during training.

**5. Convert to Torch Tensors:**
The scaled feature vectors and labels are converted into PyTorch tensors using torch.tensor. This step is necessary for compatibility with PyTorch's neural network framework.

**6. Define Neural Network Architecture:**
A neural network class NeuralNetwork is defined using PyTorch's nn.Module. It consists of two linear layers (nn.Linear) with ReLU activation function (nn.ReLU), defining the forward pass of the network.

**7. Hyperparameters Initialization:**
Hyperparameters such as the number of iterations and learning rates are defined. These parameters control the training process and optimization of the neural network:

- *"iterations"*: This hyperparameter determines the number of times the entire dataset is passed forward and backward through the neural network during the training process. It represents the number of times the model updates its parameters to minimize the loss function. Values assessed: 500, 1000, 1500.
- *"learning_rates"*:This is the hyperparameter that controls the step size at each iteration while moving toward a minimum of the loss function. It determines how much the model's parameters are adjusted in each iteration of the optimization algorithm. Values assessed: 0.001, 1.049 and 12.031.

**8. Model Training and Evaluation:**
The model is trained and evaluated using a nested loop over different combinations of iterations and learning rates. The neural network is trained using stochastic gradient descent (torch.optim.SGD) and cross-entropy loss (nn.CrossEntropyLoss).

**9. Select the Best Model and Generate Classification Report:**
The model with the highest accuracy on the test set is saved as the best model, which is used to generate predictions on the test set, and a classification report is generated using classification_report from sklearn.metrics. This report provides metrics such as precision, recall, and F1-score for each class, enabling a comprehensive evaluation of the model's performance.

**10. Results:**

(1) *Classification Report: Only-sentence Assessment's Neural Network Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.55      | 0.57   | 0.56     | 166     |
| **1 (A2 Level)**| 0.37      | 0.38   | 0.37     | 158     |
| **2 (B1 Level)**| 0.39      | 0.35   | 0.37     | 166     |
| **3 (B2 Level)**| 0.39      | 0.36   | 0.37     | 153     |
| **4 (C1 Level)**| 0.39      | 0.41   | 0.40     | 152     |
| **5 (C2 Level)**| 0.44      | 0.46   | 0.45     | 165    |
| **accuracy**    |           |        | 0.42     | 960     |
| **macro avg**   | 0.42      | 0.42   | 0.42     | 960     |
| **weighted avg**| 0.42      | 0.42   | 0.42     | 960     |

Best parameters associated with this model: {Iterations 1500, Learning Rate 1.049, Accuracy 42.29%}

(2) *Classification Report: Additional Features Assessment's Neural Network Best Model*

|                 | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| **0 (A1 Level)**| 0.75      | 0.36   | 0.49     | 153     |
| **1 (A2 Level)**| 0.43      | 0.78   | 0.56     | 156     |
| **2 (B1 Level)**| 0.56      | 0.45   | 0.50     | 153     |
| **3 (B2 Level)**| 0.50      | 0.49   | 0.49     | 173     |
| **4 (C1 Level)**| 0.42      | 0.58   | 0.49     | 166     |
| **5 (C2 Level)**| 0.64      | 0.34   | 0.44     | 159    |
| **accuracy**    |           |        | 0.50     | 960     |
| **macro avg**   | 0.55      | 0.50   | 0.50     | 960     |
| **weighted avg**| 0.55      | 0.50   | 0.49     | 960     |

Best parameters associated with this model: {Iterations 1500, Learning Rate 1.049, Accuracy 50.10%}

**Conclusion**

In comparing the performance of the neural network models for the "Only-sentence Assessment" and "Additional Features Assessment," notable differences emerge. The "Only-sentence Assessment" model achieved an overall accuracy of 42.29%, with precision, recall, and F1-scores ranging from 0.37 to 0.56 across different difficulty levels. This model demonstrated a balanced performance across most classes. Conversely, the "Additional Features Assessment" model exhibited an improved overall accuracy of 50.10%. Moreover, it excels in identifying A2 and C1 proficiency levels, with high recall scores of 0.78 and 0.58 respectively, indicating it effectively recognizes these levels but often at the cost of misclassifying other levels as A2 or C1 due to feature overlaps.

Overall, both best models utilize the same hyperparameters (Iterations: 1500, Learning Rate: 1.049), this can be attributed to the effective balance this setting achieves between training depth and step size adjustment. The choice of 1500 iterations likely provides sufficient epochs for the models to adequately converge on the data's complex patterns without overfitting, a common risk with excessive training. Meanwhile, a learning rate of 1.049 is aggressive enough to ensure rapid convergence, avoiding the slow progress associated with smaller rates like 0.001, yet it avoids the instability or divergence that can occur with excessively high rates like 12.031. This combination of parameters indicates that the models require a robust, yet cautiously quick approach to learning in order to capture and generalize the nuanced features inherent in the dataset effectively.

### 7️⃣ **Doc2Vec**

After initially training our models using traditional algorithms such as linear regression, K-nearest neighbours, and random forests, we proceeded to explore models better suited for working with text. Among several approaches, we first decided to work with the Doc2Vec model.

> Doc2Vec is a neural network-based approach that learns the distributed representation of documents. It is an unsupervised learning technique that maps each document to a fixed-length vector in a high-dimensional space. The vectors are learned in such a way that similar documents are mapped to nearby points in the vector space. (Source: [GeeksforGeeks](https://www.geeksforgeeks.org/doc2vec-in-nlp/))

Although we are familiar with Word2Vec, our initial experiments did not yield satisfactory results; hence, they are not included in this report.

Given that Doc2Vec can understand the semantic meanings of words in context more effectively than simpler models, we believe it will enhance its ability to learn document-level representations. This capability enables it to capture the context of words with greater accuracy, thereby potentially leading to higher overall accuracy of the model.

**1. Data Preparation** Prior to deploying the Doc2Vec model, essential preprocessing steps were completed to prepare French language texts, which are crucial for optimizing model performance:

*Tokenization and Cleaning:* Implemented a custom tokenizer using spaCy to:
- Lemmatize words to their base forms.
- Remove stopwords, punctuation, and placeholders like 'xx' or 'xxxx'.
- Convert all text to lowercase to ensure uniformity.
  
*Example Transformation:* 

```python
sample_sentence = "Nous sommes l'équipe SBB et nous faisons de notre mieux pour développer la meilleure machine d'apprentissage automatique pour la classification des phrases."
processed_text = spacy_tokenizer(sample_sentence)
processed_text
```

Transformed the sentence to `*équipe sbb faire mieux développer meilleur machine apprentissage automatique classification phrase*`

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

We used "saga" as it showed slightly higher results than "lbfgs" in the first training.

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

*Confusion Matrix Best model configuration*

| True/Predicted | A1   | A2   | B1   | B2   | C1   | C2   |
|----------------|------|------|------|------|------|------|
| **A1**         | 111  | 30   | 20   | 1    | 1    | 3    |
| **A2**         | 43   | 65   | 31   | 11   | 2    | 6    |
| **B1**         | 25   | 34   | 53   | 27   | 12   | 15   |
| **B2**         | 7    | 6    | 22   | 60   | 27   | 31   |
| **C1**         | 2    | 4    | 9    | 34   | 60   | 43   |
| **C2**         | 3    | 3    | 23   | 14   | 48   | 74   |


While initially including a vector size of 200 in the evaluation loop, runtime issues occurred. This configuration was computed separately, revealing a model accuracy of 40%.

The initial results suggest that the combination of Doc2Vec and Logistic Regression provides a baseline for understanding and classifying our text data. We further adding more features such as TF-IDF scores to improve the model's understanding of the text.

The integration of **TF-IDF** features with **Doc2Vec** embeddings has improved the logistic regression model for text classification:

**TF-IDF Configuration:** `TfidfVectorizer(ngram_range=(1, 2))`

Despite using the best parameters from our previous configuration for the model with a TF-IDF matrix, we encountered computational limitations using LogisticRegression(C=10, penalty='l2', solver='saga'). A default version of logistic regression was subsequently used.

- **Combined Features Testing Configuration:** (100, 8, 1, 40)
- **Combined Features Accuracy:** 44.89%

*Confusion Matrix Best model configuration with TF-IDF*

| True/Predicted | A1   | A2   | B1   | B2   | C1   | C2   |
|----------------|------|------|------|------|------|------|
| **A1**         | 119  | 34   | 12   | 0    | 0    | 1    |
| **A2**         | 60   | 67   | 26   | 2    | 2    | 1    |
| **B1**         | 34   | 45   | 50   | 19   | 10   | 8    |
| **B2**         | 17   | 7    | 21   | 52   | 32   | 24   |
| **C1**         | 10   | 3    | 10   | 21   | 63   | 45   |
| **C2**         | 12   | 5    | 9    | 30   | 29   | 80   |

From the confusion matrix, we see that the model demonstrate relatively high values along the diagonal for classes A1, B2, C1, and C2, indicating good accuracy in correctly predicting these classes. Specifically, A1 (119 correct predictions), B2 (52 correct predictions), C1 (63 correct predictions), and C2 (80 correct predictions) perform well. However, the model still struggles to distinguish the relatively close classes, for instance, A1 and A2, with 34 instances of A1 being misclassified as A2, and 60 instances of A2 being misclassified as A1. 

*Classification Report Best model configuration with TF-IDF*

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

### 8️⃣ **BERT**

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

*Results Base BERT Model*
| Epoch | Training Loss | Validation Loss | Accuracy  | F1       | Precision | Recall   |
|-------|---------------|-----------------|-----------|----------|-----------|----------|
| 1     | No log        | 1.318586        | 40.2083%  | 32.3755% | 32.7760%  | 39.3142% |
| 2     | 1.310200      | 1.162180        | 51.0417%  | 49.3079% | 50.0007%  | 50.1226% |
| 3     | 1.310200      | 1.235953        | 51.4583%  | 50.3874% | 51.5714%  | 51.0867% |


The training loss remained stable, so it is worth using a more dynamic learning rate adjustment or a longer training period to see significant changes in loss reduction.

**BERT with Increased Sequence Length**:

For this model, we adjusted some of the parameters and also implemented a learning rate scheduler that gradually reduces the learning rate as the number of epochs increases. This can help the model fine-tune its weights more effectively towards the end of training.

1) Increased sequence length: `tokenizer = BertTokenizer.from_pretrained('bert-base-cased', model_max_length=256)`

2) Other training arguments remained unchanged.

*Results BERT with Increased Sequence Length*

| Epoch | Training Loss | Validation Loss | Accuracy  | F1       | Precision | Recall   |
|-------|---------------|-----------------|-----------|----------|-----------|----------|
| 1     | No log        | 1.221369        | 48.5417%  | 47.8542% | 51.3343%  | 48.5758% |
| 2     | 1.314400      | 1.149422        | 48.7500%  | 47.8679% | 48.1385%  | 48.5678% |
| 3     | 1.314400      | 1.172957        | 50.6250%  | 49.9922% | 50.8831%  | 50.3099% |


In this case, the accuracy became lower, but by using the increased sequence length we see a decrease in the loss validation compared to our **Base BERT Model**. This can suggest that our model is generalizing better and not overfitting to the training data.

In the next step, we **increased the number of epochs to 5 epochs to monitor if there were any further improvements in the model accuracy.**

*Results BERT with Increased Sequence Length*

| Epoch | Training Loss | Validation Loss | Accuracy  | F1       | Precision | Recall   |
|-------|---------------|-----------------|-----------|----------|-----------|----------|
| 1     | No log        | 1.235162        | 48.7500%  | 47.1810% | 52.0505%  | 48.5894% |
| 2     | 1.314300      | 1.147047        | 50.6250%  | 50.0270% | 50.5915%  | 51.0392% |
| 3     | 1.314300      | 1.153481        | 50.8333%  | 49.8434% | 51.3884%  | 50.9559% |
| 4     | 0.874100      | 1.205762        | 51.0417%  | 50.7684% | 51.5135%  | 50.7474% |
| 5     | 0.874100      | 1.277318        | 50.2083%  | 49.4280% | 50.3706%  | 49.5938% |

*Training and Validation loss BERT with Increased Sequence Length*

![training loss_BERT](https://github.com/AnyaLang/SBB_ML/blob/2f1bbece203f89c17bdf049d8b5a1bcf18d99e19/Visuals/BERT%20training%20loss%20vs%20validation%20loss.png)

The training loss decreases significantly after the third epoch, indicating that the model continues to learn and improve its understanding of the training data as more epochs are processed. The validation loss does not show a clear decreasing trend; it increases slightly in the later epochs. This could suggest the beginning of **overfitting**. **The accuracy is the highest in the 4th epoch.**


**BERT with Additional Dropout**:

In addition to the previous configuration, further modifications were made:

1) To prevent overfitting or underfitting, we included the dropout rate parameter

`config = BertConfig.from_pretrained('bert-base-cased', num_labels=len(label_dict), hidden_dropout_prob=0.2)`

2) Included the warmup stage to the scheduler.

```python
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),  # 10% of training steps as warm-up
    num_training_steps=num_training_steps
)
```
3) Training arguments remain the same except that we train the model over 5 epochs


*Results for the model with max_lenght = 258*

| Epoch | Training Loss | Validation Loss | Accuracy  | F1       | Precision | Recall   |
|-------|---------------|-----------------|-----------|----------|-----------|----------|
| 1     | No log        | 1.267879        | 47.7083%  | 45.1297% | 48.1879%  | 45.8471% |
| 2     | 1.430800      | 1.203128        | 47.7083%  | 46.2310% | 50.4893%  | 47.8295% |
| 3     | 1.430800      | 1.208598        | 48.7500%  | 47.9938% | 49.4882%  | 48.9304% |
| 4     | 1.082900      | 1.232723        | 48.1250%  | 47.0514% | 49.3689%  | 48.2663% |
| 5     | 1.082900      | 1.325347        | 45.8333%  | 44.9964% | 48.4276%  | 45.4747% |

Despite improvements in the training loss, the validation loss increased as the epochs progressed, particularly notable in the fifth epoch.  The introduction of the warm-up phase has likely helped in stabilizing the training initially but has not sufficiently addressed overfitting.

We tried the same configuration for the sequence length of 128 and results were better.

*Results for the model with max_lenght = 128*

| Epoch | Training Loss | Validation Loss | Accuracy  | F1       | Precision | Recall   |
|-------|---------------|-----------------|-----------|----------|-----------|----------|
| 1     | No log        | 1.355164        | 43.4859%  | 47.8929% | 44.0591%  | 48.4929% |
| 2     | 1.420100      | 1.225194        | 48.7500%  | 47.0986% | 49.7426%  | 49.2272% |
| 3     | 1.420100      | 1.194121        | 48.9583%  | 48.0262% | 49.0184%  | 48.9477% |
| 4     | 1.080000      | 1.178910        | 51.4583%  | 50.8377% | 51.2584%  | 51.3463% |
| 5     | 1.080000      | 1.262359        | 50.0000%  | 49.0216% | 50.5802%  | 49.5725% |


**4. Results**
1) **Comparison with Traditional Classifiers**: BERT outperformed traditional machine learning classifiers (such as logistic regression) across most metrics, especially in handling nuanced language features and complex sentence structures.
2) **Training Configuration**: Models with a higher sequence length (`max_length=256`) underperformed compared to those with shorter sequences across the same training configurations. However, this does not necessarily imply that longer sequences are inherently less effective. The observed results might be specific to the chosen training setup. More comprehensive hyperparameter tuning is required to draw a definitive conclusion about the impact of sequence length on model performance.
3) **Dropout**: By increasing the time of the training, and adding additional epochs, we observed that the model seems to start overfitting. Incorporating dropout did not significantly deter overfitting in our setting. 
5) **Epochs**: Longer training periods (up to 4 epochs) generally resulted in improved accuracy
6) **Learning Rate Scheduler with Warm-up:** Introduction of a warm-up phase in the learning rate scheduler helped in stabilising early training but we did not observe substantial improvement of the model regarding scheduler preventing the model overfitting or leading to higher accuracy.

**Conclusion**: Starting with a *basic configuratio*n of `max_length = 128`, the BERT model achieved a maximum accuracy of **51.4583%**, setting a performance benchmark. BERT has outperformed traditional models with the base configuration. During the training, we have observed that shorter sequence lengths (128 tokens) demonstrated better performance across metrics compared to longer lengths. Besides, during the training of the model over 5 epochs, the performance peaked at around the fourth epoch in multiple setups. This suggests that additional changes should be made to the training of the model, e.g. learning rate management or additional regularisation. As we incorporated the learning scheduler and also the droupout to the model, the results did not improve. 

While the capabilities of this model are extensive, and further refinement of the regularization during training could be done, we decided to proceed with other models and explore other training approaches, which are more targeted towards French language. 

### 9️⃣ **FlauBERT**

**1. Model architecture**

We are using the FlauBERT model from [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/flaubert), as described:

> The FlauBERT model was proposed in the paper FlauBERT: Unsupervised Language Model Pre-training for French by Hang Le et al. It’s a transformer model pretrained using a masked language modeling (MLM) objective (like BERT).

For the FlauBERT model, one can choose from several options:

| Model name                | Number of layers | Attention Heads | Embedding Dimension | Total Parameters |
|---------------------------|------------------|-----------------|---------------------|------------------|
| flaubert-small-cased      | 6                | 8               | 512                 | 54M              |
| flaubert-base-uncased     | 12               | 12              | 768                 | 137M             |
| flaubert-base-cased       | 12               | 12              | 768                 | 138M             |
| flaubert-large-cased      | 24               | 16              | 1024                | 373M             |

We will use the FlauBERT large cased model because it offers greater depth, a more sophisticated attention mechanism, larger embedding sizes, and a higher parameter count. We believe that this model will provide us with the highest potential accuracy. Although we are aware of potential computational limitations, we have decided to deploy this model as it best aligns with the goals of our task and is specifically trained to better classify French text.

```python
model_french = 'flaubert/flaubert_large_cased'
tokenizer = FlaubertTokenizer.from_pretrained(model_french, do_lowercase=False)
```
**2. Model parameters**

The model has quite an extensive range of [parameters](https://huggingface.co/transformers/v3.2.0/model_doc/flaubert.html). For instance, default parameters of the model include language embeddings, and the `max_position_embeddings` was set to 512.  We decided to initially concentrate on hyperparameter tuning, particularly adjusting batch sizes, learning rates, and training epochs to optimise performance. If the results are not satisfactory and computational resources permit, we will delve deeper into other configurations. This may include exploring the pre-norm and post-norm settings in the transformer layers, which can affect training dynamics and model stability, different regularisation and dropout techniques etc.

**3. Hyperparameter tuning**

Upon additionally reviewing the documentation on BERT and [FlauBERT](https://huggingface.co/docs/transformers/en/model_doc/flaubert), the optimal training duration for FlauBERT is approximately four epochs, with the following hyperparameter options:

*Batch sizes:* 8, 16, 32, 64, 128

*Learning rates:* 3e-4, 1e-4, 5e-5, 3e-5

We will explore some of the listed batch sizes and the learning rates and also adjust other parameters during our training.


**4. Model training and Results**

**Evaluating the optimal parameters for training**

We will train our model only on 2 different different batch sizes to make this initial evaluation.

```python
# Parameters
learning_rates = [1e-4, 5e-5, 3e-5, 2e-5]
learning_rates = [1e-4, 5e-5, 3e-5, 2e-5]
num_epochs = 4
train_batch_sizes = [16, 32]  
eval_batch_sizes = [16, 32]   
```

*Results of the training on the batch of 16 with different learning rates:*

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


*Note on the training:*

Important to note that our first model is trained using the AdamW optimizer, which is a variant of the traditional Adam optimizer. AdamW incorporates a regularization technique known as [weight decay](https://github.com/tml-epfl/why-weight-decay), which is used in training neural networks to prevent overfitting. It functions by incorporating a term into the loss function that penalizes large weights.Besides, we also employ a Linear Learning Rate Scheduler to manage the learning rate throughout the training process. T Although this training setup does not include a warm-up phase where the learning rate would gradually ramp up before decreasing, the scheduler is configured to reduce the learning rate slightly with each training step. This gradual reduction helps in stabilizing the training as it advances.

```python
 optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=(len(train_dataset) // train_batch_size) * num_epochs)
```

*Results for different training parameters:*

Our training was interrupted, so we couldn't fully evaluate the results for a batch size of 32. However, from what we observed, a **learning rate of 5e-05 yielded the highest performance in terms of accuracy, precision, recall, and F1 score**. In the next stage, we will continue training our model using this learning rate and adjust the batch size to fully assess the results for batch 32.

We further trained the model on the batch size 32 over 4 epochs with the learning rate 5e-05. These are the results that we got:

| Epoch | Learning Rate | Average Loss     | Accuracy   | Precision | Recall   | F1 Score  |
|-------|---------------|------------------|------------|-----------|----------|-----------|
| 1/4   | 5e-5          | 0.053033780585974| 0.43125    | 0.5486596 | 0.43125  | 0.4053166 |
| 2/4   | 5e-5          | 0.040040116741632| 0.4791667  | 0.5172350 | 0.4791667| 0.4742365 |
| 3/4   | 5e-5          | 0.031953962224846| 0.5510417  | 0.5744112 | 0.5510417| 0.5477183 |
| 4/4   | 5e-5          | 0.025974183475288| 0.5739583  | 0.5810352 | 0.5739583| 0.5750711 |

The main difference between the performance of the training on the batch 16 and 32 with the same learning rate 5e-5 is the average loss. From the graph, it's clear that the average loss for the batch size of 32 is significantly lower than that for the batch size of 16 at every epoch. 

![16 and 32 batch_loss.png](https://github.com/AnyaLang/SBB_ML/blob/da060355c46c3ac513bfdad1bb53ae69696892f4/Visuals/16%20and%2032%20batch_loss.png)

![16 and 32 batch_accuracy.png](https://github.com/AnyaLang/SBB_ML/blob/d189c3932c598846fcc3212a3d20e2998d1298c8/Visuals/16%20and%2032%20batch_accuracy.png)

**Model with 5e-5 learning rate, batch 32, 6 epochs**

From our prior experience training BERT, we noticed a decrease in model accuracy after four epochs. However, in the case of FlauBERT, during previous training sessions, the model's accuracy improved from 43% to 57% over four epochs, which was substantially higher than archived by BERT. Consequently, we decided to extend the training duration to six epochs, using a batch size of 32 and see if we achieve higher accuracy with this model.


*Results with the batch 32, learning rate 5e-5 over larger number of epochs*

| Epoch | Learning Rate | Average Loss   | Accuracy   | Precision   | Recall   | F1 Score   |
|-------|---------------|----------------|------------|-------------|----------|------------|
| 1/6   | 5e-5          | 0.05631464881  | 0.4041667  | 0.4795025   | 0.4041667| 0.3879119  |
| 2/6   | 5e-5          | 0.04334156585  | 0.5333333  | 0.5451042   | 0.5333333| 0.5311008  |
| 3/6   | 5e-5          | 0.03416992826  | 0.546875   | 0.5649137   | 0.546875 | 0.5387779  |
| 4/6   | 5e-5          | 0.02753307774  | 0.6010417  | 0.6019167   | 0.6010417| 0.5942725  |
| 5/6   | 5e-5          | 0.02137682571  | 0.590625   | 0.5990724   | 0.590625 | 0.5874222  |
| 6/6   | 5e-5          | 0.01756872524  | 0.596875   | 0.6022174   | 0.596875 | 0.5978323  |

By looking at the initial results, one might conclude that 4 epochs are enough since accuracy decreased after the fourth epoch. However, extending the number of epochs allows the model to potentially generalize better if it has not yet started to overfit. This is indicated by the continued decrease in average loss and improvements in accuracy and F1 scores in your 6-epoch run.  Also, we then submitted two models on Kaggle, from the epoch 4 and 6. While the epoch 4 had higher accuracy, it provided the result of 0.573 on Kaggle. **For the model on the epoch 6th, the F1 score was higher, leading to the result of Kaggle of 0.601.** This demonstrates that relying solely on accuracy might not give a comprehensive assessment of a model's performance. Therefore, it is crucial to consider multiple metrics.

We also experimented and changed the number of epochs to 4, 6 and 8. However, for this training setting 6 epochs resulted in the highest accuracy of the model and F1 value.

**Model with a different learning rate adjustement**

Modifications for Warm-Up Phase and Learning Rate Adjustment

*Increased the Initial Learning Rate:* We start with a higher initial learning rate - 1e-4.

```python
# Initialize the optimizer with a higher initial learning rate
optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
```

*Added Warm-Up Steps:* Introduce a warm-up phase where the learning rate will linearly increase to this higher initial rate over a number of steps. A common strategy is to set the warm-up steps to 10% of the total training steps.

```python
# Scheduler with warm-up phase
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
```

*Results for 8 epochs with the adjusted learning rate, starting with 1e-04*

| Epoch | Current LR  | Training Loss | Validation Loss | Accuracy    | Precision   | Recall     | F1 Score   |
|-------|-------------|---------------|-----------------|-------------|-------------|------------|------------|
| 1/8   | 0.00009722  | 0.0629        | 0.0623          | 0.267708333 | 0.397342995 | 0.267708333| 0.221578047|
| 2/8   | 0.00008333  | 0.0508        | 0.2714          | 0.378125    | 0.476473793 | 0.378125   | 0.346038228|
| 3/8   | 0.00006944  | 0.0459        | 0.0451          | 0.45        | 0.553757879 | 0.45       | 0.425472568|
| 4/8   | 0.00005556  | 0.0348        | 0.0395          | 0.530208333 | 0.568265338 | 0.530208333| 0.522485341|
| 5/8   | 0.00004167  | 0.0249        | 0.0376          | 0.58125     | 0.594250948 | 0.58125    | 0.579418981|
| 6/8   | 0.00002778  | 0.0167        | 0.0433          | 0.6125      | 0.611882205 | 0.6125     | 0.608741828|
| 7/8   | 0.00001389  | 0.0107        | 0.0509          | 0.589583333 | 0.595456059 | 0.589583333| 0.591445696|
| 8/8   | 0.00000000  | 0.0058        | 0.0587          | 0.5875      | 0.595540066 | 0.5875     | 0.589796859|


![F1 and accuracy](https://github.com/AnyaLang/SBB_ML/blob/d189c3932c598846fcc3212a3d20e2998d1298c8/Visuals/F1%20and%20accuracy_with_adjusted%20learning_1e-04.png)

Both accuracy and F1 scores show a clear upward trend from the first to the sixth epoch, improving from 0.2677 and 0.2216 to 0.6125 and 0.6087, respectively.  Post the sixth epoch, both accuracy and F1 score plateau and then slightly decline in the final epochs (epoch 7 and 8). This stage also aligns with the increasing validation loss. 

![loss](https://github.com/AnyaLang/SBB_ML/blob/d189c3932c598846fcc3212a3d20e2998d1298c8/Visuals/loss_with_adjusted%20learning_1e-04.png)

The training loss consistently decreases from 0.0629 in the first epoch to 0.0058 in the final epoch. However, we observe that the validation loss spikes at the second epoch but decreases until the fifth epoch, achieving a low of 0.0376. However, from the sixth epoch, it begins to increase suggesting the onset of overfitting. 

*Classification report*

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| A1    | 0.74      | 0.83   | 0.78     | 166     |
| A2    | 0.57      | 0.41   | 0.48     | 158     |
| B1    | 0.51      | 0.60   | 0.55     | 166     |
| B2    | 0.58      | 0.54   | 0.56     | 153     |
| C1    | 0.55      | 0.60   | 0.57     | 152     |
| C2    | 0.71      | 0.68   | 0.70     | 165     |
|       |           |        |          |         |
| **Accuracy**  |         |        | **0.61**  | **960**   |
| **Macro Avg** | **0.61** | **0.61** | **0.61** | **960**   |
| **Weighted Avg** | **0.61** | **0.61** | **0.61** | **960**   |

*Confusion matrix*

| True \ Predicted | A1  | A2  | B1  | B2  | C1  | C2  |
|------------------|-----|-----|-----|-----|-----|-----|
| A1               | 122 | 33  | 11  | 0   | 0   | 0   |
| A2               | 26  | 77  | 51  | 3   | 1   | 0   |
| B1               | 13  | 39  | 93  | 15  | 3   | 3   |
| B2               | 0   | 4   | 20  | 92  | 29  | 8   |
| C1               | 0   | 1   | 5   | 39  | 82  | 25  |
| C2               | 0   | 0   | 4   | 14  | 49  | 98  |


The model demonstrates its strongest predictive accuracy for the A1 and C2 levels, indicating the capability to recognize the distinct linguistic features associated with the lowest and highest proficiency levels. However, it struggles with the A2 level, achieving the lowest F1-score of 0.48. For the intermediate B1 and B2 levels, the model achieves moderate F1-scores of 0.55 and 0.56, respectively. While C1 is slightly better achieving 0.57, it still indicates that the model struggles in distinguishing between closely related proficiency levels.

While the model with the adjusted learning rate demonstrated a higher accuracy score and performed better than the models before over other metrics, the submission on Kaggle provided a  score of 0.596. We also adjusted the number of epochs to 15 and lower to 4 and 6, however, the results were worse.

**Increased learning rate to 3e-4**

For the next training session, the initial learning rate was increased to 3e-4. This setting was applied over a duration of 8 epochs with a batch size of 32. Starting with a higher learning rate can potentially speed up convergence by allowing the model to make larger updates to the weights early in training. However, this approach risks overshooting the optimal points during optimization.

```python
# Initialize the optimizer with a higher initial learning rate
optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08)
    
# Scheduler with warm-up phase
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
```
While in the previous steps, we looked only at the average loss, for this step, we wanted to see the changes over the epochs in more detail.

*Results for 8 epochs with the adjusted learning rate, starting with 3e-04:*

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1 Score |
|-------|---------------|-----------------|----------|-----------|--------|----------|
| 1/8   | 0.0884        | 0.1702          | 0.1646   | 0.0646    | 0.1646 | 0.0749   |
| 2/8   | 0.1299        | 0.0601          | 0.2615   | 0.2580    | 0.2615 | 0.1882   |
| 3/8   | 0.0771        | 0.0578          | 0.3750   | 0.2530    | 0.3750 | 0.2811   |
| 4/8   | 0.0602        | 0.0439          | 0.3802   | 0.4055    | 0.3802 | 0.3560   |
| 5/8   | 0.0468        | 0.0453          | 0.4188   | 0.4033    | 0.4188 | 0.3727   |
| 6/8   | 0.0366        | 0.0395          | 0.4813   | 0.5395    | 0.4813 | 0.4491   |
| 7/8   | 0.0287        | 0.0500          | 0.4792   | 0.5071    | 0.4792 | 0.4658   |
| 8/8   | 0.0215        | 0.0536          | 0.5073   | 0.5396    | 0.5073 | 0.5053   |

*Main results:*
- The training loss consistently decreased from 0.0884 to 0.0215 across the epochs, indicating that the model is learning and improving in its predictions with each epoch.
- The validation loss started quite high at 0.1702, dropped significantly by the third epoch, and then showed minor fluctuations. 
- Accuracy, precision, recall, and F1 score all improved progressively across the epochs.

While accuracy continued to increase over the epochs, the results were not as good as for the previous model. Thus, the start with such a high learning rate might not be optimal, or it may require a longer training time.

**Model with 3e-05 learning rate, batch size 16 and a larger number of epochs**

During the training, we encountered several errors related to Memory Errors, so we decided to explore training the models with a lower batch size and different learning rates. From the previous settings, a learning rate of 5e-05 performed the best. However, we wanted to assess if the model could achieve even higher results if trained with similar parameters over a larger number of epochs, but by adopting a lower learning rate.

Therefore, we have decided to continue refining the model with this learning rate. To further explore the model's capacity, we plan to keep the batch size at 16 and adjust the learning rate to 3e-05,  while extending the training period. We believe that this will allow a more stable learning process for the model and help in mitigating the overfitting.

*Results for 15 epochs with 3e-05 learning rate and batch size 16:*

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

In the example below we demonstrate the results per epoch after the training on 15 epochs with the learning rate 3e-05 and then continuing the training on the lower learning rate for 9 epochs. This technique leverages the stability gained in earlier epochs while pushing for finer improvements in model accuracy.

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

![predictions.png](https://github.com/AnyaLang/SBB_ML/blob/d189c3932c598846fcc3212a3d20e2998d1298c8/Visuals/all_predictions.png)

**Approach we took for the 🏆 BEST MODEL 🏆 on Kaggle**

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

Observing continuous improvement, we decided that maintaining the learning rate of 2e-05 was optimal and proceeded to extend the training for 3 more epochs, however, given one issue in the code, the training extended to additional 9 epochs. Throughout this extended training period, we noticed that while the average loss consistently decreased, the accuracy improvements on our model plateaued, showing only marginal gains.

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

![predictions_best.png](https://github.com/AnyaLang/SBB_ML/blob/3d4419e9cc9ead3046b4115519c965bac65c6007/Visuals/best_graph.png)

**Conclusion**: In our project using the FlauBERT model, we explored how various training parameters, including regularization techniques, schedulers, learning rates, batch sizes, and model sizes, impacted the model's performance. We discovered that the combination of batch size and learning rate was crucial. Some settings allowed us to train the model for up to 15 epochs without it learning the training data too closely—a problem known as overfitting. However, other settings led to overfitting much earlier.

Despite spending considerable time fine-tuning the model's settings, it remains challenging to pinpoint the optimal configuration for achieving high accuracy. Nevertheless, we managed to reach a commendable 60% accuracy within just six epochs using a learning rate of 5e-05 and a batch size of 32, utilizing the AdamW optimizer configured at 3e-4. This outcome illustrates that although there is no single training strategy, carefully observing the model's response to different settings and making adjustments can lead to significant improvements. For instance, further training a previously saved model at a lower learning rate yielded even better results. Our work with the FlauBERT model was a continuous process of experimentation and refinement to identify the most effective training strategies.


### 🔟 **CamemBERT**

> CamemBERT is a state-of-the-art language model for French based on the RoBERTa architecture pretrained on the French subcorpus of the newly available multilingual corpus OSCAR. (Source: [CamemBERT](https://camembert-model.fr/))

While CamemBERT large had approximately the same number of parameters as FlauBERT, we decided to deploy this model to explore how it behaves and to see if we could improve the results obtained with FlauBERT.

Given some of the computational limitations, we have deployed the base model:

``` python
model_name = "camembert-base"
tokenizer = CamembertTokenizer.from_pretrained(model_name)
model = CamembertModel.from_pretrained(model_name)
```

![CamemBERT](https://github.com/AnyaLang/SBB_ML/blob/f2ea59e0b5f79aa55de9e48049a598a446828fb4/Visuals/CamamBERT.png)

**Model with a different learning rate adjustement**

During our first training with the base CamemBERT model, we followed a similar approach as with FlauBERT. While the model achieved lower accuracy, which we believe is mostly due to the size of the model, we noticed a much faster time for the model to reach high accuracy, at 57.71% on the third epoch, while for the FlauBERT model with the same configurations over the batch size of 32, it took 5 epochs. What is interesting is that the CamemBERT model also started overfitting much earlier than in the case of the FlauBERT model.

*Results for the Model with a different learning rate adjustement*

| Epoch | Current LR  | Training Loss | Validation Loss | Accuracy | Precision | Recall  | F1 Score |
|-------|-------------|---------------|-----------------|----------|-----------|---------|----------|
| 1/8   | 0.00009722  | 0.0468        | 0.0354          | 51.04%   | 53.36%    | 51.04%  | 48.83%   |
| 2/8   | 0.00008333  | 0.0333        | 0.0344          | 50.21%   | 52.19%    | 50.21%  | 49.93%   |
| 3/8   | 0.00006944  | 0.0252        | 0.0327          | 57.71%   | 60.71%    | 57.71%  | 57.52%   |
| 4/8   | 0.00005556  | 0.0178        | 0.0379          | 56.67%   | 56.07%    | 56.67%  | 55.98%   |
| 5/8   | 0.00004167  | 0.0118        | 0.0446          | 53.96%   | 56.81%    | 53.96%  | 54.34%   |
| 6/8   | 0.00002778  | 0.0067        | 0.0491          | 54.90%   | 57.48%    | 54.90%  | 54.97%   |
| 7/8   | 0.00001389  | 0.0039        | 0.0547          | 54.90%   | 56.90%    | 54.90%  | 55.20%   |
| 8/8   | 0.00000000  | 0.0022        | 0.0577          | 56.25%   | 58.90%    | 56.25%  | 56.44%   |

The best accuracy was achieved in the 3rd epoch, 57.71%. We see that the accuracy isn't improving consistently, suggesting some challenges in the model's ability to consistently classify new data correctly.

Both the training and validation losses generally decrease over the epochs, indicating that the model is learning and generalizing well to the validation data. However, from Epoch 5 onward, there's a noticeable increase in validation loss despite a continuing decrease in training loss. Regularisation techniques should be used to achieve better results over the training. 

**Introducing the L1 regularisation**

To address the issue of overfitting observed during the training of the CamemBERT model, we decided to introduce L1 regularisation. This regularisation technique adds a penalty equivalent to the absolute value of the magnitude of the coefficients to the loss function.

``` python
 # Add L1 regularisation
 l1_penalty = sum(p.abs().sum() for p in model.parameters())
loss += lambda_l1 * l1_penalty
```

After incorporating L1 regularisation into the training process, we observed an improvement in the model's performance in terms of generalisation. The model was less prone to overfitting, as evidenced by the more stable validation loss. However, the model achieved a lower accuracy score after introducing L1 regularisation. 

*Results with L1 Regularisation*

| Epoch | Current LR   | Training Loss | Validation Loss | Accuracy | Precision | Recall  | F1 Score | Notes                                |
|-------|--------------|---------------|-----------------|----------|-----------|---------|----------|--------------------------------------|
| 1/8   | 0.00009722   | 2.1948        | 0.0352          | 51.35%   | 53.00%    | 51.35%  | 49.33%   | Best model saved as epoch1 with F1 0.49 |
| 2/8   | 0.00008333   | 2.0949        | 0.0336          | 52.60%   | 54.18%    | 52.60%  | 52.53%   | Best model saved as epoch2 with F1 0.53 |
| 3/8   | 0.00006944   | 2.0227        | 0.0338          | 55.73%   | 57.03%    | 55.73%  | 55.75%   | Best model saved as epoch3 with F1 0.56 |
| 4/8   | 0.00005556   | 1.9692        | 0.0388          | 56.04%   | 56.60%    | 56.04%  | 55.72%   |                                      |
| 5/8   | 0.00004167   | 1.9277        | 0.0517          | 48.96%   | 53.91%    | 48.96%  | 47.61%   |                                      |
| 6/8   | 0.00002778   | 1.8938        | 0.0486          | 56.88%   | 58.83%    | 56.88%  | 56.83%   | Best model saved as epoch6 with F1 0.57 |
| 7/8   | 0.00001389   | 1.8649        | 0.0572          | 55.00%   | 57.82%    | 55.00%  | 54.98%   |                                      |
| 8/8   | 0.00000000   | 1.8430        | 0.0572          | 55.73%   | 58.16%    | 55.73%  | 55.70%   |                                      |


*Classification Report with L1 Regularisation*

| Level | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| A1    | 0.71      | 0.75   | 0.73     | 166     |
| A2    | 0.47      | 0.53   | 0.50     | 158     |
| B1    | 0.49      | 0.47   | 0.48     | 166     |
| B2    | 0.51      | 0.60   | 0.55     | 153     |
| C1    | 0.48      | 0.57   | 0.52     | 152     |
| C2    | 0.81      | 0.42   | 0.55     | 165     |
|       |           |        |          |         |
| **Accuracy** |         |        | **0.56**  | **960**   |
| **Macro Avg** | **0.58** | **0.56** | **0.56** | **960**   |
| **Weighted Avg** | **0.58** | **0.56** | **0.56** | **960**   |


**Model with a 5e-05 learning rate and 6 epochs**

*Results model with a 5e-05 learning rate 6 epochs*

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1 Score |
|-------|---------------|-----------------|----------|-----------|--------|----------|
| 1     | 0.0456        | 0.0374          | 0.4865   | 0.4749    | 0.4865 | 0.4547   |
| 2     | 0.0343        | 0.0375          | 0.4750   | 0.4930    | 0.4750 | 0.4640   |
| 3     | 0.0283        | 0.0334          | 0.5344   | 0.5370    | 0.5344 | 0.5289   |
| 4     | 0.0234        | 0.0349          | 0.5365   | 0.5622    | 0.5365 | 0.5296   |
| 5     | 0.0195        | 0.0348          | 0.5604   | 0.5760    | 0.5604 | 0.5593   |
| 6     | 0.0165        | 0.0378          | 0.5250   | 0.5470    | 0.5250 | 0.5225   |

The accuracy peaks in Epoch 5 at 56.04% and then drops in Epoch 6 to 52.50%. This fluctuation can indicate that the model may have reached its learning capacity with the current architecture and training setup.
The validation loss remains relatively stable, with a slight increase towards the last epoch. The training loss as expected reduces over the epochs.

*Confusion Matrix with a 5e-05 learning rate 6 epochs*

| Actual\Predicted | A1  | A2  | B1  | B2  | C1  | C2  |
|------------------|-----|-----|-----|-----|-----|-----|
| **A1**           | 141 | 20  | 5   | 0   | 0   | 0   |
| **A2**           | 43  | 80  | 33  | 2   | 0   | 0   |
| **B1**           | 20  | 53  | 84  | 7   | 0   | 2   |
| **B2**           | 1   | 0   | 55  | 75  | 18  | 4   |
| **C1**           | 0   | 0   | 7   | 79  | 54  | 12  |
| **C2**           | 0   | 0   | 5   | 38  | 52  | 70  |


*Classification Report with a 5e-05 learning rate 6 epochs*

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| A1    | 0.69      | 0.85   | 0.76     | 166     |
| A2    | 0.52      | 0.51   | 0.51     | 158     |
| B1    | 0.44      | 0.51   | 0.47     | 166     |
| B2    | 0.37      | 0.49   | 0.42     | 153     |
| C1    | 0.44      | 0.36   | 0.39     | 152     |
| C2    | 0.80      | 0.42   | 0.55     | 165     |
|       |           |        |          |         |
| **Accuracy** |           |        | 0.53     | 960     |
| **Macro Avg**| 0.54      | 0.52   | 0.52     | 960     |
| **Weighted Avg**| 0.55  | 0.53   | 0.52     | 960     |

 The classification report shows uneven performance across different classes as we have seen already in other models, with the model better predicting A1 and C2 classes.

**Conclusion**: The best accuracy achieved with the base CamemBERT model was 57.71% during the third epoch under a specific learning rate adjustment. Notably, this model reached a high level of accuracy more quickly than FlauBERT, although it also began to overfit earlier. Introducing L1 regularization helped address overfitting, leading to more stable validation losses and a modest improvement in model generalization, although it slightly reduced the model's peak accuracy. The parameters which we used for FlauBERT did not yield as high accuracy scores. This may be due to the smaller size of the model we deployed for CamemBERT, and also because we should have included more extensive tuning in the training, which we did not do due to the computational constraints we experienced further in the training.

## Ranking
Add by the end...

## Making predictions on a YouTube video 

It's time for a real-world test! Let's see how our model performs in a broader and more practical context. We've taken the transcript from a YouTube video designed for beginner French learners and run it through our model. Will it accurately identify A1 and A2 difficulty levels?

The specific video selected to make predictions is: [What Do French People Actually Eat? | Easy French 189](https://www.youtube.com/watch?v=p65EBC9lW9k&list=PLnazreCxpqRmpb4lGvzvCGXIXZkL3Nc27&index=4)

Example of the sentences:
`sentences = [
    "Salut les amis, bienvenue dans ce nouvel épisode.",
    "Aujourd'hui je suis dans le 11e arrondissement et je vais demander aux gens de parler de leurs habitudes alimentaires.",
    "Alors c'est parti, qu'est-ce que vous avez mangé depuis ce matin?",
]`

Since we could not access the models from the previous training due to a runtime interruption, we trained the model with a batch size of 32, for 6 epochs, and a learning rate of 5e-5 to make the predictions.

| Epoch | Learning Rate | Average Loss    | Accuracy   | Precision   | Recall    | F1 Score   |
|-------|---------------|-----------------|------------|-------------|-----------|------------|
| 1/6   | 5e-5          | 0.0604157262327 | 0.39375    | 0.5144486   | 0.39375   | 0.3701024  |
| 2/6   | 5e-5          | 0.0422514597885 | 0.4989583  | 0.5244477   | 0.4989583 | 0.4860280  |
| 3/6   | 5e-5          | 0.0340066954804 | 0.5427083  | 0.5798898   | 0.5427083 | 0.5209575  |
| 4/6   | 5e-5          | 0.0275729257846 | 0.5583333  | 0.5704371   | 0.5583333 | 0.5564640  |
| 5/6   | 5e-5          | 0.0218229048653 | 0.5677083  | 0.5970088   | 0.5677083 | 0.5620301  |
| 6/6   | 5e-5          | 0.0174353359539 | 0.5791667  | 0.5974417   | 0.5791667 | 0.5825952  |

While now this model had shown lower results compared to what we have achieved initially, still based on the results after training the model on the dataset we obtained from Kaggle, we believe our model can well predict the difficulty of the sentences. These are the results obtained:

*Predictions on the YouTube video*

![youtube_predictions.png](https://github.com/AnyaLang/SBB_ML/blob/f2ea59e0b5f79aa55de9e48049a598a446828fb4/Visuals/predictions%20YouTube%20visual.png)

The plot reveals that most sentences are classified under the A2 category, indicating that the model effectively recognizes the text as suitable for beginners. Additionally, the model identifies some sentences as more challenging, categorizing them at the B1 level. This higher classification might challenge learners, yet it also offers an opportunity to expand their vocabulary and enhance their language skills, potentially reflecting the video author's intent to promote linguistic growth."

## Streamlit Application

The previous prediction was carried out using Google Colab, a tool that many people are not familiar with. Which method could we use to allow anyone to easily utilize our model? Developing an application through Streamlit could be one way. Therefore, we decided to get to work and create the French4U app.

Given the complexity and size of our best model, we faced many difficulties in creating the Streamlit app with it. Therefore, since we wanted to create an MVP (Minimum Viable Product), we decided to implement Logistic Regression with features that achieved an accuracy of approximately 51%. Here you can find the folder with all the files that have been used for its creation: 

## Video
Add on Sunday

## Contributions

---
💟**We are excited to apply our the skills learned during this project to help others find the most suitable text for themselves to learn French or even work further on developing a more powerful model for text classification!** 💟

