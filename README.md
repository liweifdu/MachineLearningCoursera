# MachineLearningCoursera
my solution of coursera machine learning code assignments

## ex2 note
1. the `mapFeature.m` is very usefull  
   we can fit some features into **polynomial**
2. I need to know the differences between **fminunc** algorithm and gradient decent

## ex3 note
1. I need to know the differences between fmincg algorithm and fminunc algorithm
2. In the neral network inference:
   The bias adding process is **after** sigmoid activation
   ```code
   z2 = a1 * Theta1';
   a2 = [ones(m, 1) sigmoid(z2)];
   ```

## ex4 note
1. To further consider the checkNNGradients function process, and use it in my own work
2. The visual of the network, features and loss.

## ex5&6 note   
1. the choice of parameters *(regression is lambda, svm is C and sigma)* is important   
    ```code
    C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
    sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
    bst_error_val = intmax;
    ```
    then we can use for loop to update C, sigma and bst_error_val.

## ex7 note
1. computeCentroids
    we can use mean function, and find function to search the matrix of input, after match the centrol of every x(i)
    ```code
    centroids(i, :) = mean(X([find(idx == i)], :));
    ```
    
