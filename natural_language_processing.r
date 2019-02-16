# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked


# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Making a results table to collect the Accuracy, Precision, Recall and F1 of different models

# Push function, needed 
push = function(vec, item)
{
  vec = substitute(vec)
  
  if (is(item, "character")) {
    eval.parent(parse(text = paste(vec, ' <- c(', vec, ', "', item, '")', sep = '')), n=1)
  } else {
    eval.parent(parse(text = paste(vec, ' <- c(', vec, ', ', item, ')', sep = '')), n=1)
  }
}

Name = vector()
Precision = vector()
Accuracy = vector()
Recall = vector()
F1Score = vector()


# Random Forest classifier
# Fitting Random Forest Classification to the Training set
library(randomForest)
rf_classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)
# Predicting the Test set results
y_pred = predict(rf_classifier, newdata = test_set[-692])
# Making the Confusion Matrix
rf_cm = table(test_set[, 692], y_pred)
tn = rf_cm[1]
tp = rf_cm[4]
fn = rf_cm[3]
fp = rf_cm[2]
rf_accuracy = (tn + tp) / (tn + tp + fn + fp)
rf_precision = tp / (tp + fp)
rf_recall = tp / (tp + fn)
rf_f1 = 2 * rf_precision * rf_recall/(rf_precision + rf_recall)
push(Name, "Random Forest")
push(Accuracy, rf_accuracy)
push(Precision, rf_precision)
push(Recall, rf_recall)
push(F1Score, rf_f1)

#Naive Bayes Classifier
# Feature Scaling
scaled_training_set <- training_set
scaled_test_set <- test_set
scaled_training_set[-692] = scale(scaled_training_set[-692])
scaled_test_set[-692] = scale(scaled_test_set[-692])
# Fitting Naive Bayes Classifier to training set
library(e1071)
nb_classifier = naiveBayes(x = scaled_training_set[-692],
                        y = scaled_training_set$Liked)
# Predicting the Test set results
y_pred = predict(nb_classifier, newdata = scaled_test_set[-692])
# Making the Confusion Matrix
nb_cm = table(scaled_test_set[, -692], y_pred)
tn = nb_cm[1]
tp = nb_cm[4]
fn = nb_cm[3]
fp = nb_cm[2]
nb_accuracy = (tn + tp) / (tn + tp + fn + fp)
nb_precision = tp / (tp + fp)
nb_recall = tp / (tp + fn)
nb_f1 = 2 * nb_precision * nb_recall/(nb_precision + nb_recall)
push(Name, "Naive Bayes")
push(Accuracy, nb_accuracy)
push(Precision, nb_precision)
push(Recall, nb_recall)
push(F1Score, nb_f1)


#Kernel SVM
k_svm_classifier = svm(formula = Liked ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')
y_pred = predict(k_svm_classifier, newdata = test_set[-692])

# Making the Confusion Matrix
k_svm_cm = table(test_set[, 692], y_pred)
tn = k_svm_cm[1]
tp = k_svm_cm[4]
fn = k_svm_cm[3]
fp = k_svm_cm[2]
k_svm_accuracy = (tn + tp) / (tn + tp + fn + fp)
k_svm_precision = tp / (tp + fp)
k_svm_recall = tp / (tp + fn)
k_svm_f1 = 2 * k_svm_precision * k_svm_recall/(k_svm_precision + k_svm_recall)
push(Name, "Kernel SVM")
push(Accuracy, k_svm_accuracy)
push(Precision, k_svm_precision)
push(Recall, k_svm_recall)
push(F1Score, k_svm_f1)

# Decision Tree Classifier
# Feature Scaling
scaled_training_set <- training_set
scaled_test_set <- test_set
scaled_training_set[-692] = scale(scaled_training_set[-692])
scaled_test_set[-692] = scale(scaled_test_set[-692])
# Fitting Decision Tree Classifier to training set
library(rpart)
dt_classifier = rpart(formula = Liked ~ .,
                   data = training_set)
# Predicting the Test set results
y_pred = predict(dt_classifier, newdata = scaled_test_set[-692], type = 'class')

# Making the Confusion Matrix
dt_cm = table(test_set[, 692], y_pred)
tn = dt_cm[1]
tp = dt_cm[4]
fn = dt_cm[3]
fp = dt_cm[2]
dt_accuracy = (tn + tp) / (tn + tp + fn + fp)
dt_precision = tp / (tp + fp)
dt_recall = tp / (tp + fn)
dt_f1 = 2 * dt_precision * dt_recall/(dt_precision + dt_recall)
push(Name, "Decision Tree")
push(Accuracy, dt_accuracy)
push(Precision, dt_precision)
push(Recall, dt_recall)
push(F1Score, dt_f1)



Results <- data.frame(Name, Accuracy, Precision, Recall, F1Score)
