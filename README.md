# OCR
This project refers to the handwritten recognition with CNN and RNN, decode with CTC.

![demo](https://github.com/tuandoan998/OCR_IAM-dataset/blob/master/Resource/demo.png)

## Dataset
[IAM dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database)  
* [Model1] Train on WORD unit of dataset.
* [Model2] Train on LINE unit of dataset.

## Result
Test on IAM dataset:

|  Model  | Test Unit | Number of samples | CER(%) | WER(%) | 
| :-      | :-        |     :---:         |  ---:  |  ---:  |
|  WORD   | WORD      | 19289             | 10.75  | 27.46  | 
|  WORD   | LINE      | 2192              | 21.73  | 46.00  | 
|  LINE   | LINE      | 2192              | 08.32  | 28.99  | 

## Train
[Google colab]

## Usage

### Training
```
$ python3 Train.py
```

### Predict
```
$ python3 Prediction.py
```
![predict](https://github.com/tuandoan998/OCR_IAM-dataset/blob/master/Resource/test_img/Screenshot.png)

### Evaluation
```
$ python3 EvaluationWord.py
$ python3 EvaluationLine.py
```

## Model
![model1](https://github.com/tuandoan998/OCR_IAM-dataset/blob/master/Resource/word_model.png)
![model2](https://github.com/tuandoan998/OCR_IAM-dataset/blob/master/Resource/line_model.png)

## References
https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5
http://norvig.com/spell-correct.html
