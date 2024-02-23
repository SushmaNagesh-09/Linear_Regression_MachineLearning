import numpy as np

class Linear_Regression():
 
# initialing the parameters learning_rate and number of iterations    
    def __init__( self, learning_rate, no_of_iterations):
        
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        
#fitting the data to our model      
    def fit(self, X, Y):
        #number of tarining examples (m) & number of features (n)
        self.m, self.n = X.shape # number of rows(m) and columns(n)
        
        #initialing weight and bais of the model
        self.w = np.zeros(self.n)
        self.b = 0
        
        self.X = X
        self.Y = Y
        
        #implementing Gradient Descent
        
        for i in range(self.no_of_iterations):
            self.update_weights() 
        
    def update_weights(self):
        Y_prediction = self.predit(self.X)
        
        #calculating gradient
        
        dw = -(2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        
        db = - 2 * np.sum(self.Y - Y_prediction) / self.m
        
        #updating the weights
        
        self.w = self.w - self.learning_rate * dw
        
        self.b = self.b - self.learning_rate * db
         
    def predit(self, X):
        
        return X.dot(self.w)+self.b  #including dot product bcoz we are using array