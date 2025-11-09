# Delta for Spam Classifier

## ADDED Requirements

### Requirement: Data Preprocessing Pipeline
The system SHALL preprocess SMS text data for machine learning model training.

#### Scenario: Load and clean data
- **WHEN** the system loads the CSV dataset
- **THEN** it SHALL parse the label and text columns correctly
- **AND** it SHALL remove special characters, numbers, and punctuation
- **AND** it SHALL convert text to lowercase
- **AND** it SHALL tokenize the text using NLTK
- **AND** it SHALL remove stopwords

#### Scenario: Text vectorization
- **WHEN** preprocessed text is ready
- **THEN** the system SHALL convert text to numerical features using TF-IDF or Count Vectorizer
- **AND** it SHALL handle feature extraction parameters (max_features, ngram_range)

### Requirement: Model Training
The system SHALL train multiple classification models to identify spam messages.

#### Scenario: Train logistic regression model
- **WHEN** preprocessed data is available
- **THEN** the system SHALL train a logistic regression classifier
- **AND** it SHALL save the trained model to disk

#### Scenario: Train naive Bayes model
- **WHEN** preprocessed data is available
- **THEN** the system SHALL train a naive Bayes classifier
- **AND** it SHALL save the trained model to disk

#### Scenario: Train support vector machine model
- **WHEN** preprocessed data is available
- **THEN** the system SHALL train a support vector machine classifier
- **AND** it SHALL save the trained model to disk

### Requirement: Model Evaluation
The system SHALL evaluate trained models using multiple metrics and visualizations.

#### Scenario: Calculate evaluation metrics
- **WHEN** a model makes predictions on test data
- **THEN** the system SHALL calculate accuracy, precision, recall, and F1-score
- **AND** it SHALL generate a classification report

#### Scenario: Visualize model performance
- **WHEN** evaluation metrics are calculated
- **THEN** the system SHALL display a confusion matrix
- **AND** it SHALL display ROC curve and AUC score
- **AND** it SHALL compare performance across different models

### Requirement: Streamlit User Interface
The system SHALL provide an interactive web interface for users to interact with the spam classifier.

#### Scenario: Display dataset overview
- **WHEN** the user opens the Streamlit app
- **THEN** the system SHALL display dataset statistics (total messages, spam/ham distribution)
- **AND** it SHALL show sample messages

#### Scenario: Train models
- **WHEN** the user clicks the train button
- **THEN** the system SHALL train all three models
- **AND** it SHALL display training progress
- **AND** it SHALL show training completion status

#### Scenario: View model evaluation
- **WHEN** models are trained
- **THEN** the system SHALL display evaluation metrics for each model
- **AND** it SHALL show confusion matrices and ROC curves
- **AND** it SHALL allow users to compare model performance

#### Scenario: Real-time prediction
- **WHEN** the user enters a text message
- **THEN** the system SHALL predict whether it is spam or ham
- **AND** it SHALL display the prediction probability
- **AND** it SHALL highlight the prediction result

