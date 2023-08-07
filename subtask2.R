#SubTask 02
#libraries
library(data.table)
library(tidyverse)
library(plotly)
library(caTools)
library(useful)
library(tictoc)
library(readxl)
library(neuralnet)
library(Metrics)
library(ggplot2)

#Read the data
uow_data <-  read_excel("C:/Users/Thamindu/Desktop/New ml version/objective2/uow_consumption.xlsx")
#Rename the columns name
colnames(uow_data)[2] <- "6pm"
colnames(uow_data)[3] <- "7pm"
colnames(uow_data)[4] <- "8pm"

shifted_uow_data <- uow_data
shifted_uow_data <- shift.column(data = shifted_uow_data, 
                                 columns = "8pm", 
                                 len = 1, 
                                 up = FALSE,
                                 newNames = c("t1_8pm"))

shifted_uow_data <- shift.column(data = shifted_uow_data, 
                                 columns = "8pm", 
                                 len = 2, 
                                 up = FALSE,
                                 newNames = c("t2_8pm"))

shifted_uow_data <- shift.column(data = shifted_uow_data, 
                                 columns = "8pm", 
                                 len = 3, 
                                 up = FALSE,
                                 newNames = c("t3_8pm"))


shifted_uow_data <- shift.column(data = shifted_uow_data, 
                                 columns = "8pm", 
                                 len = 4, 
                                 up = FALSE,
                                 newNames = c("t4_8pm"))

shifted_uow_data <- shift.column(data = shifted_uow_data, 
                                 columns = "8pm", 
                                 len = 7, 
                                 up = FALSE,
                                 newNames = c("t7_8pm"))

#New features using 6pm and 7pm
shifted_uow_data$`Sum6_7` <- (shifted_uow_data$`6pm` + shifted_uow_data$`7pm`)

#z-score normalization
uow_data_scaled <- as.data.frame(shifted_uow_data %>% mutate_at(vars(-date), scale,
                                                                center=T))#Removing the date column
uow_data_scaled <- within(uow_data_scaled, rm(date))

#Creating logical vector that shows which samples are in training set
train <- rep(FALSE, nrow(uow_data_scaled))
train[1:380] <- TRUE

#Splitting the dataset into two parts, one is for train the data, and another is for testing.
train.data <- uow_data_scaled[train,] #For train the model
test.data <- uow_data_scaled[!train,] #For test the model

##########################################################
#NN models training and results are store in a list
nn_models <- list()
#Using formula
formula.2 <- `8pm`~t1_8pm + t2_8pm + t3_8pm + t4_8pm + t7_8pm + Sum6_7

time.start <- Sys.time()
nn_models[[1]] <- neuralnet(formula.2,
                            data = train.data,
                            hidden = c(150),#1 Hidden layer with 150 units
                            linear.output = F,
                            rep = 5,
                            act.fct = "logistic",
                            threshold = 2
)

time.end <- Sys.time()
time.training1 <- time.end - time.start

#_________________________________________________________________
time.start <- Sys.time()
nn_models[[2]] <- neuralnet(formula.2,
                            data = train.data,
                            hidden = c(150,150),
                            linear.output = F,
                            rep = 5,
                            act.fct = "logistic",
                            threshold = 2
)

time.end <- Sys.time()
time.training2 <- time.end - time.start

#_________________________________________________________________
time.start <- Sys.time()
nn_models[[3]] <- neuralnet(formula.2,
                            data = train.data,
                            hidden = c(150,150,150),
                            linear.output = F,
                            rep = 5,
                            act.fct = "logistic",
                            threshold = 2
)

time.end <- Sys.time()
time.training3 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[4]] <- neuralnet(formula.2,
                            data = train.data,
                            hidden = c(100),
                            linear.output = F,
                            rep = 5,
                            act.fct = "tanh",
                            threshold = 2
)

time.end <- Sys.time()
time.training4 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[5]] <- neuralnet(formula.2,
                             data = train.data,
                             hidden = c(150,150),
                             linear.output = F,
                             rep = 5,
                             act.fct = "tanh",
                             threshold = 2
)

time.end <- Sys.time()
time.training5 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[6]] <- neuralnet(formula.2,
                             data = train.data,
                             hidden = c(150,100,150),
                             linear.output = F,
                             rep = 5,
                             act.fct = "tanh",
                             threshold = 2
)

time.end <- Sys.time()
time.training6 <- time.end - time.start

#training scores for each nn model
train.scores <- sapply(nn_models, function(x) {min(x$result.matrix[c("error"),])})

#Print results
cat("Training Scores (Logarithmic Loss)\n with formula = `8pm`~t1_8pm + t2_8pm + t3_8pm + t4_8pm + t7_8pm + 
    Sum6_7 \n")

cat(paste(c("1 Hidden Layer, 150 Hidden Units: ",
            "2 Hidden Layers, 150 Hidden Units Each: ",
            "3 Hidden Layers, 150 with softplus Hidden Units Each: ",
            "\n1 Hidden Layers, 100 Hidden Units Each with tanh: ",
            "2 Hidden Layers, 150 Hidden Units Each with tanh: ",
            "3 Hidden Layers, 150,100,50 Hidden Units Each with tanh: "
            ),
          train.scores,
          collapse = "\n"
          )
    )
cat("\n")

#Print training times 
cat("Training times: \n")
cat(paste(c("\n1 Hidden Layer, 150 Hidden Units: ",time.training1,
            "\n2 Hidden Layers, 150 Hidden Units Each: ",time.training2,
            "\n3 Hidden Layers, 150 with softplus Hidden Units Each: ",time.training3,
            "\n\n1 Hidden Layers, 100 Hidden Units Each with tanh: ",time.training4,
            "\n2 Hidden Layers, 150 Hidden Units Each with tanh: ",time.training5,
            "\n3 Hidden Layers, 150,100,50 Hidden Units Each with tanh: ",time.training6
)
)
)
cat("\n")


#Predict using each model on the testing set
predictions <- lapply(nn_models, function(x) predict(x, test.data))


#Calculate testing scores 
test.scores <- sapply(predictions, function(x){
  rmse(test.data$`8pm`, x)
})

#Calculate testing MAE
test.mae <- sapply(predictions, function(x){
  mae(test.data$`8pm`, x)
})

#Calculate testing MAPE
test.mape <- sapply(predictions, function(x){
  mape(test.data$`8pm`, x)
})

#Calculate testing sMAPE
test.smape <- sapply(predictions, function(x){
  smape(test.data$`8pm`, x)
})


#Print results
#rmse values
cat("RMSE: \n")
cat(paste(c("1 Hidden Layer, 150 Hidden Units:",
            "2 Hidden Layers, 150 Hidden Units Each:",
            "3 Hidden Layers, 150 Hidden Units Each:",
            "1 Hidden Layers, 100 Hidden Units Each with tanh:",
            "2 Hidden Layers, 150 Hidden Units Each with tanh:",
            "3 Hidden Layers, 150,100,50 Hidden Units Each with tanh:"
            ),
          test.scores,
          collapse = "\n"
          )
    )
cat("\n")

#mae values
cat("MAE: \n")
cat(paste(c("1 Hidden Layer, 150 Hidden Units:",
            "2 Hidden Layers, 150 Hidden Units Each:",
            "3 Hidden Layers, 150 Hidden Units Each:",
            "1 Hidden Layers, 100 Hidden Units Each with tanh:",
            "2 Hidden Layers, 150 Hidden Units Each with tanh:",
            "3 Hidden Layers, 150,100,50 Hidden Units Each with tanh:"
            ),
          test.mae,
          collapse = "\n"
          )
    )
cat("\n")

#mape values
cat("MAPE: \n")
cat(paste(c("1 Hidden Layer, 150 Hidden Units:",
            "2 Hidden Layers, 150 Hidden Units Each:",
            "3 Hidden Layers, 150 Hidden Units Each:",
            "1 Hidden Layers, 100 Hidden Units Each with tanh:",
            "2 Hidden Layers, 150 Hidden Units Each with tanh:",
            "3 Hidden Layers, 150,100,50 Hidden Units Each with tanh:"
            ),
          test.mape,
          collapse = "\n"
          )
    )
cat("\n")


#smape values
cat("sMAPE: \n")
cat(paste(c("1 Hidden Layer, 150 Hidden Units:",
            "2 Hidden Layers, 150 Hidden Units Each:",
            "3 Hidden Layers, 150 Hidden Units Each:",
            "1 Hidden Layers, 100 Hidden Units Each with tanh:",
            "2 Hidden Layers, 150 Hidden Units Each with tanh:",
            "3 Hidden Layers, 150,100,50 Hidden Units Each with tanh:"
            ),
          test.smape,
          collapse = "\n"
          )
    )
cat("\n")

############ 02 i #####################
# from the first model, extracting predicted values 
predict <- predictions[[1]]

# Create a data frame
results <- data.frame(
  predicted = predict,
  actual = test.data$`8pm`
)

# Creating scatter plot
plot <- plot_ly(results, x = ~actual, y = ~predicted, type = "scatter", mode = "markers")

# Adding diagonal line in the plot
plot <- plot %>% add_trace(x = c(min(results$actual), max(results$actual)),
                           y = c(min(results$actual), max(results$actual)),
                           mode = "lines", line = list(color = "blue"))

#plot
plot


