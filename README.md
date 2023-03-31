# ml_bootcamp
ml_bootcamp

The repository contains 5 csv files and 4 ipynb files(Python notebooks) and 1 pdf containing code for neural network as I faced problem with submission of the
neural network and was instructed by my mentor to do so. The repository also contains a pdf containing a report of the project.

The report contains implementation details, some results on the performance of the algorithm on the given data and graphs for visualization.

The csv files contain the prediction of the models for the given test csv data

The ipynb

1.Linear Regression-

   To run the code, first import the training data X_train(with each row containing feature values of an example) ,Y_train 
   
   Then initialise Model=Linear_Regression()
   
   To split into train and dev ,use- X_train,Y_train,X_dev,Y_dev=Model.train_dev_split(X_train,Y_train,fraction of data to be used as train)
   
   To train use - Model.Batch_GD(X_train,Y_train,iterations=iterations, learning_rate=learning rate,L2_regularization_term=L2_regularization_term,Exp_learning_rate_decay=Exp_learning_rate_decay_termFeature_Scaling=Either Z_score_standardization or min_max_normalization,validation=True or False,X_dev=X_dev,Y_dev=Y_dev)
   
   Then to print results use- Model.Results(MSE_learning_curve=True or False,R2_learning_curve=True or False,Single_feature_vs_label=True or False,Table_showing_predicted_vs_actual=True or False)
   
2. Polynomial Regression-

  To run the code, first import the training data X_train(with each row containing feature values of an example) ,Y_train 
   
   Then initialise Model=Polynomial_Regression()
   
   To add polynomial features use Model.add_features(X_train,degree)
   
   To split into train and dev ,use- X_train,Y_train,X_dev,Y_dev=Model.train_dev_split(X_train,Y_train,fraction of data to be used as train)
   
   To train using batch gradient descent use-Model.Batch_GD(X_train,Y_train,iterations=iterations,learning_rate=learning rate,L2_regularization_term=L2_regularization_term,Exp_learning_rate_decay=Exp_learning_rate_decay_term,Feature_Scaling=Either Z_score_standardization or min_max_normalization,validation=True or False,X_dev=X_dev,Y_dev=Y_dev)
   
   To train using mini batch gradient descent use-Model.mini_batch_GD(X_train,Y_train,iterations=iterations,learning_rate=learning rate,L2_regularization_term=L2_regularization_term,Exp_learning_rate_decay=Exp_learning_rate_decay_term,mini_batch_size=mini_batch_size,Feature_Scaling=Either Z_score_standardization or min_max_normalization,validation=True or False,X_dev=X_dev,Y_dev=Y_dev)
  
  Then to print results use- Model.Results(MSE_learning_curve=True or False,R2_learning_curve=True or False,Single_feature_vs_label=True or False,Table_showing_predicted_vs_actual=True or False)
   
 3.K Nearest Neighbours-
   
   To run the code, first import the training data X_train(with each row containing feature values of an example) ,Y_train 
   
   Then initialise Model=KNN(value of k to use)
   
   To split into train and dev ,use- X_train,Y_train,X_dev,Y_dev=Model.train_dev_split(X_train,Y_train,fraction of data to be used as train)
   
   To predict for a test data use- Model.predict(X_train,Y_train,X_test,Feature_Scaling=Z_score_standardization or min_max_normalization)
   
   To find accuracy use- Model.accuracy(Y_test,Y_pred)
   
 4.Logistic Regression
  
  To run the code, first import the training data X_train(with each row containing feature values of an example) ,Y_train 
   
   Then initialise Model=Logistic_Regression()
   
   To split into train and dev ,use- X_train,Y_train,X_dev,Y_dev=Model.train_dev_split(X_train,Y_train,fraction of data to be used as train)
   
   To train the model using batch gradeint descent use-Model.Batch_GD(X_train,Y_train,iterations=iterations,learning_rate=learning rate,L2_regularization_term=L2_regularization_term,Exp_learning_rate_decay=Exp_learning_rate_decay_term,Feature_Scaling=Either Z_score_standardization or min_max_normalization,validation=True or False,X_dev=X_dev,Y_dev=Y_dev)
   
   To train the model using mini batch gradient descent use-Model.mini_batch_GD(X_train,Y_train,epochs=epochs,learning_rate=learning rate,L2_regularization_term=L2_regularization_term,Exp_learning_rate_decay=Exp_learning_rate_decay_term,mini_batch_size=mini_batch_size,Feature_Scaling=Either Z_score_standardization or min_max_normalization,validation=True or False,X_dev=X_dev,Y_dev=Y_dev)
   
   To print results use- Model.Results(Cost_learning_curve=True or False,Accuracy_learning_curve=True or False,Table_showing_predicted_vs_actual=True or False)
   
   To predict use Model.predict(X_test)
   
  5.Neural Network
   
   To run the code, first import the training data X_train(with each row containing feature values of an example) ,Y_train 
   
   Then initialise Model=Neural_Network(List containing number of neurons in each hidden layer)
   
   To split into train and dev ,use- X_train,Y_train,X_dev,Y_dev=Model.train_dev_split(X_train,Y_train,fraction of data to be used as train)
   
   To train using batch gradient descent use-Model.Batch_GD(X_train,Y_train,iterations=iterations,learning_rate=learning rate,L2_regularization_term=L2_regularization_term,Exp_learning_rate_decay=Exp_learning_rate_decay_term,Feature_Scaling=Either Z_score_standardization or min_max_normalization,dropout=True or False,dropout_probability=Probability of dropping,validation=True or False,X_dev=X_dev,Y_dev=Y_dev)
   
   To train using mini batch gradient descent use- Model.mini_batch_GD(X_train,Y_train,epochs=epochs,learning_rate=learning rate,L2_regularization_term=L2_regularization_term,Exp_learning_rate_decay=Exp_learning_rate_decay_term,Feature_Scaling=Either Z_score_standardization or min_max_normalization,dropout=True or False,mini_batch_size=mini_batch_size,dropout_probability=Probability of dropping,validation=True or False,X_dev=X_dev,Y_dev=Y_dev)
   
   To train using Adam optimization use-Model.Adam(X_train,Y_train,epochs=epochs,learning_rate=learning rate,L2_regularization_term=L2_regularization_term,beta1=beta1,beta2=beta2,epsilon=epsilon,Exp_learning_rate_decay=Exp_learning_rate_decay_term,Feature_Scaling=Either Z_score_standardization or min_max_normalization,dropout=True or False,mini_batch_size=mini_batch_size,dropout_probability=Probability of dropping,validation=True or False,X_dev=X_dev,Y_dev=Y_dev)
   
   To print results use- Model.Results(Cost_learning_curve=True or False,Accuracy_learning_curve=True or False,Table_showing_predicted_vs_actual=True or False)
   
   To predict use Model.predict(X_test,normalized=True or False)
