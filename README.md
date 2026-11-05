# Handwritten Digit Classification

## a. Problem statement

The task is to identify which digit, from 0 through 9, is written in a small grayscale image. Each image is stored as 64 pixel intensities from an 8 by 8 grid. The work compares five classifiers on one shared, stratified hold-out test set and does not retrain those models when the test file is scored again.

## b. Dataset description

The dataset is the UCI Optical Recognition of Handwritten Digits collection, using the 8 by 8 copy distributed with scikit-learn (`sklearn.datasets.load_digits`). This is a published subset of the UCI dataset: 1,797 images, which is above the 500-instance minimum, and 64 features, which is above the 12-feature minimum.

Source: Alpaydin, E. and Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

- Each feature is named `pixel_row_column` and takes an integer value from 0 to 16. There are no missing values.
- The target column is `digit`. The ten classes are nearly balanced: the smallest class (digit 8) has 174 images and the largest (digit 3) has 183.
- The split is stratified 80/20 with `random_state=42`: 1,437 training rows and 360 test rows. The 360 test rows are saved as `test_data.csv`, including the `digit` column.
- Hyperparameters were chosen by cross-validation on the training rows only. The test rows were used once, for the table below.
- Logistic regression, k-nearest neighbors, and the support vector machine use a `StandardScaler` fitted on the training rows. The decision tree is not scaled, because its splits do not depend on feature scale. Multinomial naive Bayes is also not scaled, because the pixel values are already non-negative and scaling would create negative numbers that this model cannot accept.
- Precision, recall, and F1 are weighted averages over the ten digits. AUC is the weighted one-vs-rest score from `predict_proba`. MCC is the multiclass Matthews correlation coefficient.

## c. Github Repository Link

https://github.com/2026ak05243/ml-assignment2-digits

## d. Models used

All five models were trained on the same training split and scored on the same 360-row test split.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9750 | 0.9991 | 0.9752 | 0.9750 | 0.9750 | 0.9722 |
| Decision Tree | 0.8333 | 0.9074 | 0.8332 | 0.8333 | 0.8321 | 0.8151 |
| KNN | 0.9639 | 0.9951 | 0.9648 | 0.9639 | 0.9636 | 0.9600 |
| Naive Bayes | 0.8889 | 0.9914 | 0.8890 | 0.8889 | 0.8876 | 0.8768 |
| Support Vector Machine | 0.9750 | 0.9995 | 0.9759 | 0.9750 | 0.9749 | 0.9723 |

Selected settings, from the training-set search: logistic regression uses `C=1` with balanced class weights; the decision tree uses `max_depth=16` and `min_samples_leaf=1` with balanced class weights; k-nearest neighbors uses 5 neighbors and uniform weights; naive Bayes is multinomial with `alpha=0.5`; the support vector machine uses an RBF kernel, `C=1`, and `gamma=scale`, with balanced class weights.

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strong linear model on this pixel data. Held-out accuracy is 0.9750 and AUC is 0.9991. Training accuracy is 0.9993, so the drop on unseen digits is small. The repeated mistake is between 8 and 1: three 8s are read as 1, and two 1s are read as 8. At 8 by 8 resolution those two shapes share a thin vertical stroke. |
| Decision Tree | Weakest of the five, and the clearest case of overfitting. Training accuracy is 1.0000 while test accuracy is 0.8333, test F1 is 0.8321, and MCC is 0.8151. Cross-validation still chose a deep tree (`max_depth=16`, leaves of one sample), and that tree memorizes training images instead of learning a stroke. The largest test confusions are 1 read as 2 (6 rows) and 8 read as 1 (5 rows). |
| KNN | Close to the two leading models, with test accuracy 0.9639 and F1 0.9636. Training accuracy is 0.9840, so the gap is modest. Five equally weighted neighbors beat distance-weighted neighbors in the training search. The largest single confusion is 8 read as 1 (3 rows). Scaling is used because the vote depends on Euclidean distance. |
| Naive Bayes | Test accuracy is 0.8889, but AUC remains high at 0.9914. Training accuracy is 0.9054, close to the test score, so this is underfit rather than memorization. Multinomial naive Bayes treats the 64 pixels as independent given the digit. Neighboring pixels in a stroke are not independent, and that hurts the final label more than the class ranking. The largest confusions are 8 read as 1 (6), 1 read as 2 (5), and 5 read as 9 (5). |
| Support Vector Machine | Best cross-validated training accuracy of the search (0.9798) and the best held-out ranking. Test accuracy ties logistic regression at 0.9750. AUC is 0.9995, precision is 0.9759, and MCC is 0.9723, all slightly above logistic regression. Weighted F1 is 0.9749, which is 0.0001 below logistic regression and not a meaningful gap. Training accuracy is 0.9972, so the margin fits the digits without the tree's perfect training score. Remaining mistakes are mostly 8 read as 1 and 9 read as 7, two rows each. |
| Overall, Winner for your dataset? | Support Vector Machine. Logistic regression matches its accuracy and is 0.0001 higher on F1. The tie is broken by AUC, then MCC: the RBF SVM leads on both, and it also has the higher precision. |
