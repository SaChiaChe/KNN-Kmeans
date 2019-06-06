# KNN & Kmeans
Practice of KNN and Kmeans.

## How to run

```
python KNN.py TrainData TestData
```
```
python Kmeans.py TrainData
```

## Description

### KNN.py

KNN is a clustering algoithm, an exsample will be predicted by it's k nearest neighbors uniformly. I experimented with different k, and plot a graph at the end.

Besides KNN, another very simple clustering algorithm is also implemented, named "Uniform", it votes with all the examples in the training data, with different weight, closer points give bigger weight, and vice versa. 

### Kmeans.py

Kmeans is also a clustering algorithm, it starts by randomly pick k examples, and start with these examples as centers, and the training data will be optimally partitioned by finding the closest center to it. Then recalculate the center by finding the mean point of all examples of a cluster. Repeat the above 2 steps until converge.

## Built With

* Python 3.6.8 :: Anaconda custom (64-bit)

## Authors

* **SaKaTetsu** - *Initial work* - [SaKaTetsu](https://github.com/SaKaTetsu)