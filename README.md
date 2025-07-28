# Customer Review Classification

## Project Overview
This project aims to predict customer ratings (1-5) for Singapore Airlines based on text reviews using Natural Language Processing (NLP) and machine learning techniques. The goal is to provide airlines with a tool to understand customer feedback, mitigate oversight, and improve services by reconstructing ratings from textual data.

## Dataset
The project utilizes the **Singapore Airlines Reviews** dataset from Kaggle, containing approximately 10,000 anonymized customer reviews with corresponding ratings (1-5). The dataset provides insights into customer satisfaction and perceptions of the airline's services.

- **Source**: [Singapore Airlines Reviews on Kaggle](https://www.kaggle.com/datasets/singapore-airlines-reviews)
- **Split**: The dataset is divided into 80% training and 20% validation sets using `train_test_split` with `test_size=0.2`.

## Methodology
The project involves preprocessing text reviews and applying various word embedding and machine learning models to predict ratings. The following methods were compared:

1. **One Hot Encoding + Random Forest**:
   - **Challenges**: High dimensionality, sparse representation, and lack of contextual information.
   - **Outcome**: Unable to train due to large vocabulary size.

2. **Bag of Words + Random Forest**:
   - **Parameters**: `max_features=500`, `n_estimators=100`, `criterion='entropy'`.
   - **Advantages**: Considers word frequency.
   - **Disadvantages**: Similar to One Hot Encoding (high dimensionality, no context).
   - **Accuracy**: 62.77%.

3. **Continuous Bag of Words (CBOW) + XGBoost**:
   - Uses word embeddings to capture context.
   - **Details**: Input as a 2D array (window_size, 2), projected and summed to generate embeddings.
   - **Outcome**: Limited details provided due to computational constraints.

4. **Pre-trained Word2Vec + SVM/XGBoost/Neural Network**:
   - **SVM Parameters**: `kernel='rbf'`.
   - **Accuracy**: 69.78% (SVM with Word2Vec).
   - **Confusion Matrix**: Indicates model confusion between ratings 1-2 and 4-5.

## Data Preprocessing
- **Cleaning**: Removed noise and irrelevant information from reviews.
- **NLP Techniques**:
  - **Tokenization**: Breaking text into tokens.
  - **Stemming/Lemmatization**: Reducing words to their root/base form.
  - **Part-of-Speech Tagging**: Assigning grammatical categories to words.
  - **Stopwords Removal**: Using `nltk` to filter out common words.
  - **TF-IDF Vectorization**: Applied using `sklearn.feature_extraction.text.TfidfVectorizer`.

## Models and Libraries
The project uses the following Python libraries:
- `nltk`: For text processing (tokenization, stopwords).
- `scikit-learn`: For machine learning models (`RandomForestClassifier`, `SVC`, `train_test_split`, `accuracy_score`).
- `numpy`: For numerical computations.
- `xgboost`: For gradient boosting.
- `matplotlib` and `seaborn`: For visualization (e.g., confusion matrix heatmap).

## Model Performance
- **Bag of Words + Random Forest**: 62.77% accuracy.
- **Word2Vec + SVM**: 69.78% accuracy, with noted confusion between ratings 1-2 and 4-5.
- **PCA Visualization**: Attempted for word embeddings but yielded poor results due to computational limitations.

## Challenges
- High dimensionality in One Hot Encoding made training infeasible.
- Models struggled to distinguish between closely related ratings (e.g., 1 vs. 2, 4 vs. 5).
- Computational constraints limited the effectiveness of PCA visualization and CBOW training.
