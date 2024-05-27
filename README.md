# Machine Learning and Deep Learning Exercises

Welcome to the Machine Learning and Deep Learning practice exercises! This repository contains weekly exercises designed to help you practice various classification algorithms using TensorFlow and deep learning techniques. Each week focuses on a different algorithm, including regularization and optimization techniques, and culminates in Natural Language Processing (NLP) tasks.

## Weekly Exercises

### Week 1: Logistic Regression
**Exercise:** Implement a binary logistic regression model to classify whether a patient has diabetes using the Pima Indians Diabetes Database.

**Dataset:** [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

**Tasks:**
1. Load and preprocess the dataset (handle missing values, normalization).
2. Build a logistic regression model using TensorFlow.
3. Train the model with different regularization strengths and evaluate its performance using metrics such as accuracy, precision, recall, and F1-score.
4. Visualize the training process (loss curve) and compare performance with and without regularization.

### Week 2: k-Nearest Neighbors (k-NN)
**Exercise:** Implement the k-NN algorithm to classify handwritten digits using the MNIST dataset. Use TensorFlow to manage data but implement k-NN from scratch.

**Dataset:** [MNIST](https://www.tensorflow.org/datasets/catalog/mnist)

**Tasks:**
1. Load the MNIST dataset and preprocess it.
2. Implement the k-NN algorithm from scratch.
3. Tune the hyperparameter \( k \) to find the optimal number of neighbors.
4. Use TensorFlow functions to handle the dataset and calculate distances.
5. Evaluate the model on the test set and analyze its performance.

### Week 3: Decision Trees
**Exercise:** Build a decision tree classifier to predict species of iris flowers using the Iris dataset. Implement the model using TensorFlow Decision Forests.

**Dataset:** [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)

**Tasks:**
1. Load and preprocess the Iris dataset.
2. Use TensorFlow Decision Forests to build the decision tree model with pruning.
3. Train and evaluate the model using appropriate metrics.
4. Visualize the decision tree and discuss feature importance.

### Week 4: Support Vector Machine (SVM)
**Exercise:** Implement an SVM to classify types of cancer using the Breast Cancer Wisconsin (Diagnostic) dataset. Use TensorFlow for data handling and Scikit-learn for the SVM implementation.

**Dataset:** [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

**Tasks:**
1. Load and preprocess the dataset.
2. Implement an SVM using Scikit-learn.
3. Perform hyperparameter tuning (C, gamma) to optimize model performance.
4. Use TensorFlow to handle data preprocessing and visualization.
5. Train the SVM model and evaluate its performance.

### Week 5: Naive Bayes with Smoothing
**Exercise:** Build a Naive Bayes classifier to classify SMS messages as spam or not spam using the SMS Spam Collection dataset. Implement the model using TensorFlow and Scikit-learn.

**Dataset:** [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

**Tasks:**
1. Load and preprocess the dataset (tokenization, TF-IDF).
2. Implement a Naive Bayes classifier using Scikit-learn.
3. Apply Laplace smoothing to handle zero probabilities.
4. Use TensorFlow for data handling and visualization.
5. Train and evaluate the model on the dataset.

### Week 6: Random Forest
**Exercise:** Build a Random Forest classifier to predict whether a passenger survived the Titanic disaster. Use TensorFlow Decision Forests.

**Dataset:** [Titanic Dataset](https://www.kaggle.com/c/titanic/data)

**Tasks:**
1. Load and preprocess the Titanic dataset (handle missing values, encoding categorical variables).
2. Implement a Random Forest classifier using TensorFlow Decision Forests.
3. Perform hyperparameter tuning (number of trees, max depth) to optimize model performance.
4. Train and evaluate the model using appropriate metrics.
5. Visualize the feature importance and discuss the results.

### Week 7: Neural Networks
**Exercise:** Implement a simple neural network to classify fashion items using the Fashion MNIST dataset. Use TensorFlow/Keras for the implementation.

**Dataset:** [Fashion MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist)

**Tasks:**
1. Load and preprocess the Fashion MNIST dataset.
2. Build a neural network with dropout and L2 regularization using TensorFlow/Keras.
3. Train the model and evaluate its performance using accuracy and loss metrics.
4. Visualize the training process and the performance of the model on test data.

### Week 8: Convolutional Neural Networks (CNNs)
**Exercise:** Implement a CNN to classify images in the CIFAR-10 dataset. Use TensorFlow/Keras for the implementation.

**Dataset:** [CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10)

**Tasks:**
1. Load and preprocess the CIFAR-10 dataset.
2. Build a CNN using TensorFlow/Keras.
3. Apply data augmentation techniques (rotation, flipping, cropping) to the training data.
4. Train the model and evaluate its performance using accuracy and loss metrics.
5. Visualize the training process and some of the learned filters.

### Week 9: Recurrent Neural Networks (RNNs)
**Exercise:** Implement an RNN to classify sequences from the IMDB movie review dataset as positive or negative. Use TensorFlow/Keras for the implementation.

**Dataset:** [IMDB Movie Reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews)

**Tasks:**
1. Load and preprocess the IMDB dataset (tokenization, padding).
2. Build an RNN (LSTM/GRU) using TensorFlow/Keras.
3. Train the model and evaluate its performance using accuracy and loss metrics.
4. Visualize the training process and discuss the results.

### Week 10: Ensemble Methods
**Exercise:** Implement an ensemble model combining different classifiers (e.g., logistic regression, decision trees, and SVM) to classify the Wine dataset. Use TensorFlow for data handling and Scikit-learn for the ensemble methods.

**Dataset:** [Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine)

**Tasks:**
1. Load and preprocess the Wine dataset.
2. Implement individual classifiers using Scikit-learn.
3. Combine the classifiers using ensemble techniques (e.g., voting, stacking) with regularization.
4. Train and evaluate the ensemble model.

### Week 11: Transfer Learning
**Exercise:** Use a pre-trained model (e.g., VGG16, ResNet) to classify images in the Cats vs. Dogs dataset. Use TensorFlow/Keras for the implementation.

**Dataset:** [Cats vs. Dogs](https://www.kaggle.com/c/dogs-vs-cats/data)

**Tasks:**
1. Load and preprocess the Cats vs. Dogs dataset.
2. Use a pre-trained model and fine-tune it for the classification task.
3. Apply data augmentation and regularization techniques to improve performance.
4. Train the model and evaluate its performance.
5. Visualize the training process and discuss the results.

### Week 12: Generative Adversarial Networks (GANs)
**Exercise:** Implement a GAN to generate new images based on the MNIST dataset. Use TensorFlow/Keras for the implementation.

**Dataset:** [MNIST](https://www.tensorflow.org/datasets/catalog/mnist)

**Tasks:**
1. Load and preprocess the MNIST dataset.
2. Build and train a GAN using TensorFlow/Keras with gradient penalty for stability.
3. Generate new images and evaluate the quality of generated images.
4. Visualize the training process and the generated images.

### Week 13: Natural Language Processing (NLP) with BERT
**Exercise:** Fine-tune a BERT model for text classification on the IMDB movie review dataset. Use TensorFlow/Keras for the implementation.

**Dataset:** [IMDB Movie Reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews)

**Tasks:**
1. Load and preprocess the IMDB dataset (tokenization using BERT tokenizer).
2. Fine-tune a pre-trained BERT model for the classification task using TensorFlow/Keras.
3. Apply techniques like learning rate scheduling and dropout for regularization.
4. Train the model and evaluate its performance using accuracy and loss metrics.
5. Visualize the training process and discuss the results.

## Getting Started

1. Clone this repository to your local machine.
2. Set up your Python environment with the required libraries (TensorFlow, Scikit-learn, etc.).
3. Follow the weekly exercises and complete the tasks.
4. Share your results and insights with your friends for discussion and further learning.

## Requirements

- Python 3.6+
- TensorFlow 2.0+
- Scikit-learn
- Matplotlib (for visualization)
- Jupyter Notebook (optional but recommended)

## Contributing

Feel free to contribute by adding more exercises, improving existing ones, or fixing any issues. Fork this repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Happy learning and coding!
