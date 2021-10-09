from PIL import Image
import numpy as np

img = Image.open('cars/resized_train/1.jpg')
train_x = np.asarray(img).reshape(270000,1)  #normalde tek bir örnek için train_x (300,300,3) shape'inde bir matrix. onu (300*300*3,1) shape'ine çeviriyoruz, yani unroll ediyoruz. 
                                                   

for i in range(2,201):  #training set 200 resimden oluşuyor.
    i = str(i)
    img = Image.open('cars/resized_train/'+i+'.jpg')
    numpydata = np.asarray(img).reshape(270000,1)
    train_x = np.hstack((train_x,numpydata))
 
train_x = train_x / 255   #standardizasyon                           #sonuçta train_x (300*300*3,m) shape'inde bir matrix'e dönüşüyor.

img = Image.open('cars/resized_test/1.jpg')
test_x = np.asarray(img).reshape(270000,1)


for i in range(2,51):  #test set 50 resimden oluşuyor.
    print(i)
    i = str(i)
    img = Image.open('cars/resized_test/'+i+'.jpg')
    numpydata = np.asarray(img).reshape(270000,1)
    test_x = np.hstack((test_x,numpydata))


test_x = test_x/255  #standardizasyon


#train setteki resimlerin ilk 100 tanesi araba resmi, kalan 100 tanesi ise alakasız resimler. bu yüzden buna uygun (1,200) shape'inde train_y matrisi yaratıyoruz.
train_y = np.hstack((np.ones((1,100)),np.zeros((1,100))))

#test setteki resimlerin ilk 25 tanesi alakasız resim, kalan 25 tanesi ise araba resmi. bu yüzden (1,50) shape'ine sahip uygun matris yaratıyoruz.
test_y = np.hstack((np.zeros((1,25)),np.ones((1,25))))

#train_x, train_y, test_x, test_y matrislerini istediğimiz formata getirdikten sonra ML yapacak fonksiyonları oluşturuyoruz.

#=================================================================================================================================================================

#sigmoid fonksiyonunu hem propagate hem de predict fonksiyonunda kullanıyoruz.
#bizim için en önemli özelliği predict fonksiyonundaki işlevi. çünkü sigmoid fonksiyonu ile
#(1,m) shapeine sahip bir matris elde ediyoruz ve bu matrisin elemanları eğer 0.5'ten büyükse
#predict fonksiyonu ile direkt 1, değilse 0 kabul ediliyor.

# z = np.dot(w.T,X) + b 

def sigmoid(z):             
    s = 1/(1+np.exp(-z))
    return s

  
#w bir sütun matris olmalı, b ise float 0 olmalı.

def initialize(dim):
    w = np.zeros((dim,1))
    b = float(0)
    return w,b


#burada forward ve back propagation yapıyoruz. aslında özünde yaptığımız şey tamamen chain rule'dan ibaret. 

def propagate(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -(1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)
    
    grads = {"dw":dw, "db":db}
    return grads, cost

#yukardaki propagate fonksiyonunu aşağıdaki gradient descent algoritmasında for loop içine yazıyoruz ki her iterationda bize dw ve db hesaplasın. 
#optimizasyonu gradient descent algoritmasıyla yapıyoruz. 

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


 
#predict fonksiyonu ile sigmoid fonksiyonun outputu olan A matrisine bakılarak Y_prediction tahmin ediliyor.

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


#son olarak model() fonksiyonuyla yukardaki tüm fonksiyonları tek bir çatı altında topluyoruz.  

def model(X_train,Y_train,X_test,Y_test,iter=2000,lrate=0.5,print_cost = False):
    w,b = initialize(X_train.shape[0])
    params,grads,costs = optimize(w,b,X_train,Y_train,iter,lrate,print_cost)
    w = params["w"]
    b = params["b"]
    Y_predict_test = predict(w,b,X_test)
    Y_predict_train = predict(w,b,X_train)
    
    if print_cost:
        print(f'train accuracy: {100-np.mean(np.abs(Y_predict_train - Y_train))*100}%')  #train accuracy hesabı.
        print(f'test accuracy: {100-np.mean(np.abs(Y_predict_test - Y_test))*100}%')     #test accuracy hesabı.




# model(train_x, train_y, test_x, test_y, iter= 250, lrate= 0.0009, print_cost = True)


# 250 iteration ve 0.0009 learning rate için yukardaki kodu çalıştırdığımızda, 
  
# train accuracy: 97.5% 
# test accuracy:  80.0% 

#şeklinde bir output alıyoruz ki böyle basit bir algoritma için iyi bir oran.
#iteration'ı 250'den daha fazla yapmamamızın sebebi overfitting olmasını engellemek. 
#örneğin 2000 yaptık diyelim, bu durumda train accuracy 100% oluyor fakat test accuracy'den kaybediyoruz.
#ilerleyen zamanlarda bunu önlemek için cost function içine regularization ifadesi ekleyeceğiz.