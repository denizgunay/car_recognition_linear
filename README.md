# car_recognition_linear
This is a simple study to understand how linear regression works. Dataset only consists of 200 training data and 50 test data, and all images have 300 x 300 pixels. 

### How?
What we do here first is to unroll all images then train linear regression model by using forward and back prop. For 250 iteration and learning rate = 0.0009, we see that 

* train accuracy = 97.5%
* test accuracy = 80.0% 

### About result

As can be seen, there is a big difference between train and test accuracy. Since our train and test sets are shuffled and have the same distribution, we can say that the reason for that situation arises from high variance problem. In this case, we can use regularization methods such as L2 and early stop to deal with that. But in this example, regularization is not applied.    


