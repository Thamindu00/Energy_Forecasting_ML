#libraries
#Subtask 01
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
uow_data <-  read_excel("C:/Users/Thamindu/Desktop/New ml version/objective2/uow_consumption.xlsx")#location of the uow_consumption.xlsx 

#summary(uow_data)
#print(uow_data)

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

head(shifted_uow_data)

############## 01 c##########################
#z-score normalization
uow_data_scaled <- as.data.frame(shifted_uow_data %>% mutate_at(vars(-date), scale,
                                                                center=T))#Removing the date column
uow_data_scaled <- within(uow_data_scaled, rm(date))
head(uow_data_scaled)
#Creating logical vector that shows which samples are in training set
train <- rep(FALSE, nrow(uow_data_scaled))
train[1:380] <- TRUE

#Splitting the dataset into two parts, one is for train the data, and another is for testing.
train.data <- uow_data_scaled[train,] #For train the model
dim(train.data)

test.data <- uow_data_scaled[!train,] #For test the model
dim(test.data)

############## 01 d##########################
#NN models training and results are store in a list
nn_models <- list()
#Using formula
formula.1 <- `8pm`~t1_8pm + t2_8pm + t3_8pm + t4_8pm + t7_8pm
 
time.start <- Sys.time()
nn_models[[1]] <- neuralnet(formula.1,
                         data = train.data,
                         hidden = c(50),#1 Hidden layer with 50 units
                         linear.output = FALSE,
                         rep = 5,
                         act.fct = "logistic",
                         threshold = 2
                         )

time.end <- Sys.time()
time.training1 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[2]] <- neuralnet(formula.1,
                            data = train.data,
                            hidden = c(100),
                            linear.output = FALSE,
                            rep = 5,
                            act.fct = "logistic",
                            threshold = 2
                            )

time.end <- Sys.time()
time.training2 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[3]] <- neuralnet(formula.1,
                            data = train.data,
                            hidden = c(150),
                            linear.output = FALSE,
                            rep = 5,
                            act.fct = "logistic",
                            threshold = 2
                            )

time.end <- Sys.time()
time.training3 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[4]] <- neuralnet(formula.1,
                            data = train.data,
                            hidden = c(50,50),
                            linear.output = FALSE,
                            rep = 5,
                            act.fct = "logistic",
                            threshold = 2
                            )

time.end <- Sys.time()
time.training4 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[5]] <- neuralnet(formula.1,
                            data = train.data,
                            hidden = c(100,100),
                            linear.output = FALSE,
                            rep = 5,
                            act.fct = "logistic",
                            threshold = 2
                            )

time.end <- Sys.time()
time.training5 <- time.end - time.start
#_________________________________________________________________

#_________________________________________________________________
time.start <- Sys.time()
nn_models[[6]] <- neuralnet(formula.1,
                            data = train.data,
                            hidden = c(150,150),
                            linear.output = FALSE,
                            rep = 5,
                            act.fct = "logistic",
                            threshold = 2
                            )

time.end <- Sys.time()
time.training6 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[7]] <- neuralnet(formula.1,
                            data = train.data,
                            hidden = c(100,100,100),
                            linear.output = FALSE,
                            rep = 5,
                            act.fct = "logistic",
                            threshold = 2
                            )

time.end <- Sys.time()
time.training7 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[8]] <- neuralnet(formula.1,
                            data = train.data,
                            hidden = c(150,150,150),
                            linear.output = FALSE,
                            rep = 5,
                            act.fct = "logistic",
                            threshold = 2
                            )

time.end <- Sys.time()
time.training8 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[9]] <- neuralnet(formula.1,
                            data = train.data,
                            hidden = c(100),
                            linear.output = FALSE,
                            rep = 5,
                            act.fct = "tanh",
                            threshold = 2
                            )

time.end <- Sys.time()
time.training9 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[10]] <- neuralnet(formula.1,
                            data = train.data,
                            hidden = c(150,150),
                            linear.output = FALSE,
                            rep = 5,
                            act.fct = "tanh",
                            threshold = 2
                            )

time.end <- Sys.time()
time.training10 <- time.end - time.start
#_________________________________________________________________
time.start <- Sys.time()
nn_models[[11]] <- neuralnet(formula.1,
                             data = train.data,
                             hidden = c(150,100,50),
                             linear.output = FALSE,
                             rep = 5,
                             act.fct = "tanh",
                             threshold = 2
                            )

time.end <- Sys.time()
time.training11 <- time.end - time.start

#nn_models
#training scores for each nn model
train.scores <- sapply(nn_models, function(x) {min(x$result.matrix[c("error"),])})

#Print results
cat("Training Scores (Logarithmic Loss)\n with formula = `8pm`~ t1_8pm + t2_8pm + t3_8pm + t4_8pm +
    t7_8pm \n")
cat(paste(c("1 Hidden Layer, 50 Hidden Units: ",
            "1 Hidden Layer, 100 Hidden Units: ",
            "1 Hidden Layer, 150 Hidden Units: ",
            "2 Hidden Layers, 50 Hidden Units Each: ",
            "2 Hidden Layers, 100 Hidden Units Each: ",
            "2 Hidden Layers, 150 Hidden Units Each: ",
            "3 Hidden Layers, 100 Hidden Units Each: ",
            "3 Hidden Layers, 150 Hidden Units Each: ",
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
cat("Training Times: \n")
cat(paste(c("1 Hidden Layer, 50 Hidden Units: ",time.training1,
            "\n1 Hidden Layer, 100 Hidden Units: ",time.training2,
            "\n1 Hidden Layer, 150 Hidden Units: ",time.training3,
            "\n2 Hidden Layers, 50 Hidden Units Each: ",time.training4,
            "\n2 Hidden Layers, 100 Hidden Units Each: ",time.training5,
            "\n2 Hidden Layers, 150 Hidden Units Each: ",time.training6,
            "\n3 Hidden Layers, 100 Hidden Units Each: ",time.training7,
            "\n3 Hidden Layers, 150 Hidden Units Each: ",time.training8,
            "\n\n1 Hidden Layers, 100 Hidden Units Each with tanh: ",time.training9,
            "\n2 Hidden Layers, 150 Hidden Units Each with tanh: ",time.training10,
            "\n3 Hidden Layers, 150,100,50 Hidden Units Each with tanh: ",time.training11
            )
          )
    )
cat("\n")
##################################################################
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
cat(paste(c("1 Hidden Layer, 50 Hidden Units: ",
            "1 Hidden Layer, 100 Hidden Units: ",
            "1 Hidden Layer, 150 Hidden Units: ",
            "2 Hidden Layers, 50 Hidden Units Each: ",
            "2 Hidden Layers, 100 Hidden Units Each: ",
            "2 Hidden Layers, 150 Hidden Units Each: ",
            "3 Hidden Layers, 100 Hidden Units Each: ",
            "3 Hidden Layers, 150 Hidden Units Each: ",
            "\n1 Hidden Layers, 100 Hidden Units Each with tanh: ",
            "2 Hidden Layers, 150 Hidden Units Each with tanh: ",
            "3 Hidden Layers, 150,100,50 Hidden Units Each with tanh: "
            ),
          test.scores,
          collapse = "\n"
          )
    )
cat("\n")

#mae values
cat("MAE: \n")
cat(paste(c("1 Hidden Layer, 50 Hidden Units: ",
            "1 Hidden Layer, 100 Hidden Units: ",
            "1 Hidden Layer, 150 Hidden Units: ",
            "2 Hidden Layers, 50 Hidden Units Each: ",
            "2 Hidden Layers, 100 Hidden Units Each: ",
            "2 Hidden Layers, 150 Hidden Units Each: ",
            "3 Hidden Layers, 100 Hidden Units Each: ",
            "3 Hidden Layers, 150 Hidden Units Each: ",
            "\n1 Hidden Layers, 100 Hidden Units Each with tanh: ",
            "2 Hidden Layers, 150 Hidden Units Each with tanh: ",
            "3 Hidden Layers, 150,100,50 Hidden Units Each with tanh: "
            ),
            test.mae,
          collapse = "\n"
          )
    )
cat("\n")

#mape values
cat("MAPE: \n")
cat(paste(c("1 Hidden Layer, 50 Hidden Units: ",
            "1 Hidden Layer, 100 Hidden Units: ",
            "1 Hidden Layer, 150 Hidden Units: ",
            "2 Hidden Layers, 50 Hidden Units Each: ",
            "2 Hidden Layers, 100 Hidden Units Each: ",
            "2 Hidden Layers, 150 Hidden Units Each: ",
            "3 Hidden Layers, 100 Hidden Units Each: ",
            "3 Hidden Layers, 150 Hidden Units Each: ",
            "\n1 Hidden Layers, 100 Hidden Units Each with tanh: ",
            "2 Hidden Layers, 150 Hidden Units Each with tanh: ",
            "3 Hidden Layers, 150,100,50 Hidden Units Each with tanh: "
            ),
          test.mape,
          collapse = "\n"
          )
    )
cat("\n")


#smape values
cat("sMAPE: \n")
cat(paste(c("1 Hidden Layer, 50 Hidden Units: ",
            "1 Hidden Layer, 100 Hidden Units: ",
            "1 Hidden Layer, 150 Hidden Units: ",
            "2 Hidden Layers, 50 Hidden Units Each: ",
            "2 Hidden Layers, 100 Hidden Units Each: ",
            "2 Hidden Layers, 150 Hidden Units Each: ",
            "3 Hidden Layers, 100 Hidden Units Each: ",
            "3 Hidden Layers, 150 Hidden Units Each: ",
            "\n1 Hidden Layers, 100 Hidden Units Each with tanh: ",
            "2 Hidden Layers, 150 Hidden Units Each with tanh: ",
            "3 Hidden Layers, 150,100,50 Hidden Units Each with tanh: "
            ),
          test.smape,
          collapse = "\n"
          )
    )
cat("\n")

