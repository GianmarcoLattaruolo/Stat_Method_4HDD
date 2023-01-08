## ---------------------------------------------------------------------------------------------
## Sparse Methods for News Classification
## Statistical Methods for High Dimensional Data project code
## Authors: Anna Badalyan, Andrii Kliachkin, Gianmarco Lattaruolo, Francesco Sartori
## ---------------------------------------------------------------------------------------------
# installing and Loading the required libraries
install.packages('quanteda')
install.packages('quanteda.textplots')
install.packages('glmnet')
install.packages('doMC')
install.packages('sparseSVM')
install.packages('doParallel')

library('doParallel')
library('quanteda')
library('quanteda.textplots')
library('glmnet')
library('doMC')
library('sparseSVM')

# the dataset file is available at google drive
id <- "1t_bpJX7PC0dqZ5pEmaxM8wmDJteYmOXB" # google file ID
news <- read.csv(sprintf("https://docs.google.com/uc?id=%s&export=download", id), row.names=NULL, sep = ';')
nrow(news)

# function to prepre the dataframe, removes punctuation, symbols, numbers, site urls, separators and
# separator characters, stop words, lowercases and stemmatizes the words. Then a document frequency matrix
# with words that appear less than docfreq = 8 times is built
prepare_dfm <- function(src, fit=TRUE, words_to_include=NULL, docfreq = 8){
  # does the preprocessing steps, returns a dfm
  # fit=TRUE trims based on frequency
  # fit=FALSE makes the features=words_to_include
  tokens <- quanteda::tokens(src,remove_punct = TRUE,
            remove_symbols = TRUE,
            remove_numbers = TRUE,
            remove_url = TRUE,)
  dfm <- quanteda::dfm(tokens, tolower = TRUE)
  dfm <- quanteda::dfm_remove(dfm,stopwords('english'))
  dfm <- quanteda::dfm_wordstem(dfm)
  if(fit) {
    dfm <- quanteda::dfm_trim(dfm, min_docfreq = docfreq)
  } else {
    dfm <- quanteda::dfm_match(dfm, words_to_include)
  }
  return(dfm)
}

dfm_train <- prepare_dfm(data_train$title)
dfm_test <- prepare_dfm(data_test$title, fit=FALSE, words_to_include = featnames(dfm_train))

# tf-idf, calculated on train
# weight the matrices with tf
x_train <- quanteda::dfm_weight(dfm_train, scheme='prop')
x_test <- quanteda::dfm_weight(dfm_test, scheme='prop')
# calculate idfs on train
idfs <- docfreq(dfm_train, scheme='inverse')
# weight by idfs
x_train <- quanteda::dfm_weight(x_train, weights=idfs, force=TRUE)
x_test <- quanteda::dfm_weight(x_test, weights=idfs, force=TRUE)

x_train <- as(x_train,"sparseMatrix")
x_test <- as(x_test,"sparseMatrix")
y_train <- as.factor(news$topic)[mask]
y_test <- as.factor(news$topic)[-mask]

## ---------------------------------------------------------------------------------------------
## Data Exploration
## ---------------------------------------------------------------------------------------------

# plotting word frequencies
dfm_news <- prepare_dfm(news$title)
x_news <- quanteda::dfm_weight(dfm_news, scheme='boolean')
y_news <- as.factor(news$topic)
plot(sort(colSums(x_news)),log='y', main = 'word log-frequencies (sorted)', type = 's', xlab = 'words', ylab = ' occurences of the word')

# plotting a word cloud based on 150 most common words in each topic
par(mfrow = c(1,1))
docvars(x_news, 'topic') <- news$topic

for (i in c('TECHNOLOGY', 'BUSINESS', 'WORLD' ,'SCIENCE' ,'HEALTH', 'ENTERTAINMENT' ,'SPORTS', 'NATION')){
  topic.todisplay <- i
  textplot_wordcloud(
    dfm_subset(x_news, topic==topic.todisplay),
    color=rev(RColorBrewer::brewer.pal(10, "RdBu")),
    max_words = 150
  )
  print(i)
}

## ---------------------------------------------------------------------------------------------
## Building Models
## ---------------------------------------------------------------------------------------------

#We remove the useless variables to free some RAM space.
rm(list=ls()[! ls() %in% c("x_train","x_test","y_train","y_test")])
gc()

## ---------------------------------------------------------------------------------------------
## Multinomial Logistic Regression with cross validation for alpha and lambda
## ---------------------------------------------------------------------------------------------

# activate parallel computing
registerDoMC()

#  fit a generic elastic net tunig the values of lambda and alpha
foldid <- sample(1:3, size = length(y_train), replace = TRUE)
cv1 <- cv.glmnet(x_train, y_train, family="multinomial", type.measure="class", nfolds=3, parallel=TRUE, foldid = foldid, alpha = 1)
cv.5 <- cv.glmnet(x_train, y_train, family="multinomial", type.measure="class", nfolds=3, parallel=TRUE, foldid = foldid, alpha = 0.5)
cv0 <- cv.glmnet(x_train, y_train, family="multinomial", type.measure="class", nfolds=3, parallel=TRUE, foldid = foldid, alpha = 0)
cv.8 <- cv.glmnet(x_train, y_train, family="multinomial",type.measure="class", nfolds=3, parallel=TRUE, foldid = foldid, alpha = 0.8)
cv.2 <- cv.glmnet(x_train, y_train, family="multinomial",type.measure="class", nfolds=3, parallel=TRUE, foldid = foldid, alpha = 0.2)

# check which model has the lowest avarage cross validation error
c(min(cv0$cvm), min(cv.2$cvm), min(cv.5$cvm), min(cv.8$cvm), min(cv1$cvm))

# plot the results for the alpha - 1, 0.2 and 0
par(mfrow = c(2,2))
plot(cv1); plot(cv.5); plot(cv0);
plot(log(cv1$lambda) , cv1$cvm , pch = 19, col = "red",
xlab = "log(Lambda)", ylab = cv1$name)
points(log(cv.2$lambda), cv.5$cvm, pch = 19, col = "grey")
points(log(cv0$lambda) , cv0$cvm , pch = 19, col = "blue")
legend("topleft", legend = c("alpha= 1", "alpha= .2", "alpha 0"),pch = 19, col = c("red","grey","blue"))

#making predictions with alpha = 0
pred.train <- predict(cv0, newx = x_train, type="class")
pred.test <- predict(cv0, newx=x_test, type="class")
print(mean(y_train==pred.train))
print(mean(y_test==pred.test))

# function to plot the worldcloud with coefficients
plot_coefs <- function(coefs, features) {
	dfm.coefs <- quanteda::dfm(tokens(''))
	dfm.coefs <- dfm_match(dfm.coefs, features) + 1
	abs.coefs <- abs(as.vector(coefs)[-1])
	names(abs.coefs) <- features
	dfm.coefs <- dfm_weight(dfm.coefs, weights=abs.coefs,force=TRUE)
	textplot_wordcloud(
		dfm.coefs,
		color=rev(RColorBrewer::brewer.pal(10, "RdBu")),
    max_words = 100)
}

# plot the coefficients for health topic
plot_coefs(coef(cv1)$HEALTH, colnames(x_train))

## ---------------------------------------------------------------------------------------------
## Sparse SVM
## ---------------------------------------------------------------------------------------------

# make a validadtion set
size <- floor(0.6 * nrow(x_train))
mask <- sample(seq_len(nrow(x_train)), size=size)
x_val <- x_train[-mask,]
x_train <- x_train[mask, ]
y_val <- y_train[-mask]
y_train <- y_train[mask]

# the function to create balanced dataset for each topic
stratify_dataset <- function(x, y, topic){
    ind_pos <- which(y==topic)
    n <- length(ind_pos)
    mask_pos <- sample(ind_pos, size=n)
    x_pos <- x[mask_pos,]
    y_pos <- rep(1, n)
    
    ind_rest <- which(y!=topic)
    mask_neg <- sample(ind_rest, size=n)
    x_neg <- x_train[mask_neg,]
    y_neg <- rep(0, n)
    
    x_svm <- rbind(x_pos, x_neg)
    y_svm <- c(y_pos, y_neg)
    res <- list(x_svm, y_svm)
    return (res)
}

# training procedure for sparseSVM, the best lambda is chosen using the validation set
topics <- unique(y_train)
ovr.models <- vector('list', length(topics))
ovr.acc <- vector('list', length(topics))
best.lm <- vector('list', length(topics))

for(i in 1:length(topics)) {
 #stratify for one v. rest
 topic <- topics[i]
 res_train <- stratify_dataset(x_train, y_train, topic)
 x_train_svm <- res_train[[1]]
 y_train_svm <- res_train[[2]]

 res_val <- stratify_dataset(x_val, y_val, topic)
 x_val_svm <- res_val[[1]]
 y_val_svm <- res_val[[2]]
 
 message(sprintf("\nStarted training on %s", topic))
 model.svm <- sparseSVM(as.matrix(x_train_svm), y_train_svm, nlambda = 7, lambda = c(0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 1/dim(x_train_svm)[1]))
 ovr.models[[i]] <- model.svm

 # check the best lambda on validation
 pred_val <- predict(model.svm, as.matrix(x_val_svm), lambda=model.svm$lambda, type='class')
 err <- as.matrix(colSums(abs(pred_val-y_val_svm)))
 lm_index <- which.min(err)
 best.lm[[i]] <- 1

 rm(x_train_svm)
 gc()
}

# the function to make predictions using the coefficients
predict_with_coef <- function(object, lambda_index, x_test) {
  class = predict(object, x_test, lambda=object$lambda[lambda_index], type='class')
  coefs = predict(object, x_test, lambda=object$lambda[lambda_index], type='coefficients')
  coefs <- as.matrix(coefs)
  # add an interesept to the matrix
  x_test_int <- cbind(matrix(1, nrow(x_test), 1), x_test)
  preds <- x_test_int %*% coefs
  # make sure that the positive class is > 0, negative class is < 0
  if (((preds[1] <= 0) & (class[1]==0)) | ((preds[1] > 0) & (class[1]==1))){
    }  else{
      preds = -preds
    }

  return(preds)
}

# make predictions over all classes by choosing the highest value 
df = data.frame(matrix(nrow = dim(x_test)[1], ncol = length(topics)))
colnames(df) <- topics

for (i in 1:length(topics)) {
  topic <- topics[i]
  model <- ovr.models[[i]]
  lm.index <- best.lm[[i]]

  # making predictions with the best lambda
  preds <- predict_with_coef(model, lm.index, x_test)
  df[i] <- as.numeric(preds)
}

predictions <- as.factor(colnames(df)[apply(df,1,which.max)])
# check the accuracy
print(mean(y_test == predictions))

## ---------------------------------------------------------------------------------------------
## Grouped Lasso
## ---------------------------------------------------------------------------------------------

# Using the glmnet package we fit a grouped lasso with ten values of lambda and alpha equal to 1.
registerDoParallel()
cv <- glmnet(x_train, y_train, family="multinomial", type.measure="class",nlambda = 10, parallel=TRUE,type.multinomial='grouped')
cv$lambda
# make predictions
pred.train <- predict(cv, newx = x_train, type="class")
pred.test <- predict(cv, newx=x_test, type="class")

for (x in c('s0','s1','s2','s3','s4','s5','s6','s7','s8','s9')) {
  print(x)
  print(mean(y_train==pred.train[,x]))
  print(mean(y_test==pred.test[,x]))
}

# The best model gives the accuracy on the test set 79% and 88% on train
plot(cv,xvar="lambda")
plot(cv,xvar="dev")
plot(cv,xvar="norm")

## ---------------------------------------------------------------------------------------------
## Word Embeddings from BERT
## ---------------------------------------------------------------------------------------------

# load the dataset wirh word embeddings
options(timeout=300)
id <- "1kgfBvHHJ-uWnth6GKP89m9dJhof3fMUu" # google file ID
news <- read.csv(sprintf("https://docs.google.com/uc?id=%s&export=download&confirm=t", id))
nrow(news)

# split on train and test
set.seed(123)
size <- floor(0.8 * nrow(news))
mask <- sample(seq_len(nrow(news)), size=size)
x_train <- news[mask, ]
x_test <- news[-mask, ]
x_train <- as.matrix(x_train[,-769])
x_test <- as.matrix(x_test[,-769])

y_train <- as.factor(news$topic)[mask]
y_test <- as.factor(news$topic)[-mask]

# remove the useless variables to free some RAM space.
rm(list=ls()[! ls() %in% c("x_train","x_test","y_train","y_test")])
gc()

## Fit multinomial logistic regression
registerDoParallel()

foldid <- sample(1:3, size = length(y_train), replace = TRUE)
cv1 <- cv.glmnet(x_train, y_train, family="multinomial", type.measure="class", nfolds=3, parallel=TRUE, foldid = foldid, alpha = 1)
cv0 <- cv.glmnet(x_train, y_train, family="multinomial", type.measure="class", nfolds=3, parallel=TRUE, foldid = foldid, alpha = 0)
cv05 <- cv.glmnet(x_train, y_train, family="multinomial", type.measure="class", nfolds=3, parallel=TRUE, foldid = foldid, alpha = 0.5)

# chose the best model
c(min(cv0$cvm), min(cv05$cvm), min(cv1$cvm))

# accuracy by the best model
pred.train <- predict(cv0, newx = x_train, type="class")
pred.test <- predict(cv0, newx=x_test, type="class")
print(mean(y_train==pred.train))
print(mean(y_test==pred.test))