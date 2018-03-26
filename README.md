# kaggle_titanic
Several scripts and notebook related to the Titanic competition on Kaggle

### Exploratory data analysis (EDA)

**see file exploratory_analysis_Kaggle_submission_20180307.ipynb**

**also published as a [kaggle public kernel](https://www.kaggle.com/pepacz/titanic-dataset-exploratory-analysis)**

Exploratory analysis of the dataset, containing mainly various plots along with takeaways from these plots.

### Kaggle competition submission - RandomForestClassifier with scikit-learn pipeline

**see file submission_RandomForestClassifier_20180307a.py**

**also published as a [kaggle public kernel](https://www.kaggle.com/pepacz/randomforestclassifier-with-sklearn-pipeline)**

Competition submission. The script contains scikit-learn pipeline with some custom transformers for data preprcessing. I tried to use the TPOT package in order to automatize the model selection part of the ML process. This part of the code is commented out as the TPOT is not currently supported in the Kaggle environment (as of 2018-03-26). My impression from the TPOT result is rather mixed as the classifier reaches 44th percentile among submissions. On the other hand there are many overfitted submissions in the competition in my opinion.

### Precision-recall comparison for various scikit-learn estimators

**see file precision_recall_comparison_20180326.ipynb**

This notebook shows how various estimators perform on the Titanic dataset in terms of their precision-recall tradeoff. Purpose of this notebook is rather educational and the estimators' performance should be taken with reserve as the estimators' hyperparameters are not fine-tuned.
