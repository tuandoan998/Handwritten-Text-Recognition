# OCR
This project refers to the handwritten recognition with CNN and RNN, decode with CTC.

![demo](https://github.com/tuandoan998/OCR_IAM-dataset/blob/master/Resource/demo.png)
[Demo on youtube](https://youtu.be/kILhJXcR7To)

# Update
This [repo](https://github.com/tuandoan998/vietnamese-htr) has better model and higher accuracy 

## Dataset
[IAM dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database)  
* [Model1 - word_model.png] Train on WORD unit of dataset.
* [Model2 - line_model.png] Train on LINE unit of dataset.
* Split data:
https://www.openslr.org/56/
https://drive.google.com/drive/u/0/folders/1xqek060j9_zVWGY80m33qSCrZf3scqUy

## Result
Test on IAM dataset:

|  Model  | Test Unit | CER(%) | WER(%) | 
| :-      | :-        |  ---:  |  ---:  |
|  WORD   | WORD      | 17.20  | 37.78  | 

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
![predict](https://github.com/tuandoan998/OCR_IAM-dataset/blob/master/Resource/predict.png)

### Evaluation
```
$ python3 EvaluationWord.py
$ python3 EvaluationLine.py
```

## References
https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5
http://norvig.com/spell-correct.html
