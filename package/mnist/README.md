# Gradle Kotlin 

## requirements
kotlin( >= 1.1.2 )
CUDA( == 8.0 )
cuDNN( == 6.5 or 7.0 )
Gradle( >= 4.0 )


## run each example
### MnistSingleLayer in Train Mode
```
$ gradle MnistSingleLayer -Pargs="train"
```
### MnistSingleLayer in Predict Mode
```
$ gradle MnistSingleLayer -Pargs="predict"
```


### MLPMnistTwoLayerExample
```
$ gradle MLPMnistTwoLayerExample
```
etc.
