# Naive-Bayes-Classification
Text Classification of Scientific articles using naive bayes classifier.
The data set used contains sentences from the abstract and introduction of scientific articles that come from three different domains(300 articles from each domain):
1. PLoS Computational Biology (PLOS)
2. The machine learning repository on arXiv (ARXIV)
3. The psychology journal Judgment and Decision Making (JDM).

Text classification of articles is achieved in 2 steps:
1. Pre-processing of articles : articles are converted to features to be used by naive bayes classifier.
2. Classification of articles : Carried out in 2 phases
    a. Training the classifier : naive bayes classifier reads training data and learns the parameters.
    b. Testing the classifier  : trained classifier is tested using testing data and class of the article being tested is predicted.

Accuracy of the classifier is computed by comparing the actual class and predicted class label of each article in testing data.
