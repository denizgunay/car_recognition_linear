from PIL import Image
import numpy as np

img = Image.open('cars/resized_train/1.jpg') 
train_x = np.asarray(img).reshape(270000,1)  #for one example, train_x is a matrix with shape (300,300,3). therefore, we are giving a new shape which is (300*300*3,1)  
                                                   

for i in range(2,201):  #training set consists of 200 images.
    i = str(i)
    img = Image.open('cars/resized_train/'+i+'.jpg')
    numpydata = np.asarray(img).reshape(270000,1)
    train_x = np.hstack((train_x,numpydata))
 
train_x = train_x / 255   #standardization                           #now, train_x is a matrix with shape (300*300*3,m)

img = Image.open('cars/resized_test/1.jpg')
test_x = np.asarray(img).reshape(270000,1)


for i in range(2,51):  #test set consists of 50 images.
    print(i)
    i = str(i)
    img = Image.open('cars/resized_test/'+i+'.jpg')
    numpydata = np.asarray(img).reshape(270000,1)
    test_x = np.hstack((test_x,numpydata))


test_x = test_x/255  #standardization


#first 100 images in the training set are car images, and last 100 are irrelevant images. therefore, we create a train_y matrix with shape (1,200) accordingly.
train_y = np.hstack((np.ones((1,100)),np.zeros((1,100))))
 
#first 25 images in the test set are irrelevant images, and last 25 are car images. therefore, we create a test_y matrix with shape (1,50) accordingly. 
test_y = np.hstack((np.zeros((1,25)),np.ones((1,25))))

#after creating train_x, train_y, test_x, test_y matrices, now we can make the functions we need.

#=================================================================================================================================================================


# z = np.dot(w.T,X) + b 

def sigmoid(z):             
    s = 1/(1+np.exp(-z))
    return s

  
#w should be a column matrix, and b should be float 0.

def initialize(dim):
    w = np.zeros((dim,1))
    b = float(0)
    return w,b


#here, we do calculations for the forward and back propagate. 

def propagate(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -(1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)
    
    grads = {"dw":dw, "db":db}
    return grads, cost

#also, for every iteration we use propagate() in the for loop below while calculating dw and db.  

def optimize(w,b,X,Y,iter=100,lrate=0.009,print_cost=False):
    
    costs = []
    for i in range(iter):
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - lrate*dw
        b = b - lrate*db
        
        if i%100 == 0:
            costs.append(cost)
            if print_cost:
                print(f'Cost after iteration {i}: {cost}')
    
    params = {"w":w,"b":b}
    grads = {"dw":dw,"db":db}
    
    return params,grads,costs


#predict() function estimates the result by looking at the matrix A which is the output of the sigmoid function.  

def predict(w,b,X):
    m = X.shape[1]
    Y_predict = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X) + b)
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_predict[0,i] = 1
        else:
            Y_predict[0,i] = 0
            
    return Y_predict


#finally, we gather all the above functions under one roof with the model() function.

def model(X_train,Y_train,X_test,Y_test,iter=2000,lrate=0.5,print_cost = False):
    w,b = initialize(X_train.shape[0])
    params,grads,costs = optimize(w,b,X_train,Y_train,iter,lrate,print_cost)
    w = params["w"]
    b = params["b"]
    Y_predict_test = predict(w,b,X_test)
    Y_predict_train = predict(w,b,X_train)
    
    if print_cost:
        print(f'train accuracy: {100-np.mean(np.abs(Y_predict_train - Y_train))*100}%')  #calculation for train accuracy.
        print(f'test accuracy: {100-np.mean(np.abs(Y_predict_test - Y_test))*100}%')     #calculation for test accuracy.




# model(train_x, train_y, test_x, test_y, iter= 250, lrate= 0.0009, print_cost = True)

# OUTPUT; 
  
# train accuracy: 97.5% 
# test accuracy:  80.0% 

#NOTE: This result show us there is a high variance problem. 
