# Step 1: Install and load necessary libraries
install.packages(c("tm", "textclean", "e1071", "wordcloud", "caret", "ggplot2"))
library(tm)
library(textclean)
library(e1071)
library(wordcloud)
library(caret)
library(ggplot2)

# Step 2: Sample Data (You can replace with your own CSV)
data <- data.frame(
  text = c(
    "I love this product! It's amazing.",
    "Terrible experience, would not recommend.",
    "Absolutely fantastic service!",
    "Worst app I've ever used.",
    "Good value for the money.",
    "This was a waste of time and money."
  ),
  sentiment = c("positive", "negative", "positive", "negative", "positive", "negative")
)

# Step 3: Text Preprocessing Function
clean_text <- function(text) {
  corpus <- VCorpus(VectorSource(text))
  corpus <- tm_map(corpus, content_transformer(replace_contraction))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeWords, stopwords("en"))
  corpus <- tm_map(corpus, stripWhitespace)
  return(corpus)
}

# Apply preprocessing
corpus <- clean_text(data$text)

# Step 4: Create Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.99)  # keep important terms
dataset <- as.data.frame(as.matrix(dtm))
dataset$sentiment <- as.factor(data$sentiment)

# Step 5: Train/Test Split
set.seed(123)
train_index <- createDataPartition(dataset$sentiment, p = 0.8, list = FALSE)
train_data <- dataset[train_index, ]
test_data <- dataset[-train_index, ]

# Step 6: Model Training using Naive Bayes
model <- naiveBayes(sentiment ~ ., data = train_data)

# Step 7: Prediction & Evaluation
predictions <- predict(model, test_data)
conf_matrix <- confusionMatrix(predictions, test_data$sentiment)
print(conf_matrix)

# Step 8: Insights - Visualizing Word Cloud
wordcloud(corpus, min.freq = 1, max.words = 100, colors = brewer.pal(8, "Dark2"))

# Step 9: Plotting Accuracy
accuracy <- conf_matrix$overall['Accuracy']
barplot(accuracy, beside = TRUE, col = "blue", main = "Model Accuracy")

