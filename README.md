# CSB_P2

Project 2 for CSB 320.

run: 
```
conda env create 
conda activate csb320-p2
```

Then simply run all Jupyter cells in 
    - analyses_fake_news_kaggle_english.ipynb 
    and 
    - analyses_fake_news_spanish_jpposadas.ipynb 
Allocate ample time and resources for these scripts to run. (This time can be decreased if using CuML or decreasing grid search hyperparameters / cv folds)

## File Descriptions

### lib.py

Imports all modules, defines some basic functions for use in library

### NLP_algorithms.py

Contains presets for configurations of each learning model

### analyses_fake_news_kaggle_english.ipynb (and .html)

Evaluation and report of machine learning methods on Diabetes dataset

### analyses_fake_news_spanish_jpposadas.ipynb (and .html)

Evaluation and report of machine learning methods on Wisconsin Breast Cancer dataset

## Discussions, conclusions, and reflections of results from two reports

For English dataset from Kaggle:

- How well does TF-IDF work for feature extraction?
    TF-IDF worked very well for feature extraction. The limit of 5000 features seemed to be effective and the vector did not take a riduculous amount of memory nor computation time.
- What challenges did you encounter in text preprocessing?
    However, preprocessing did indeed consuming a very large amount of computational resources, which was one of the most difficult challenges during this project.
- How did the Random Forest model perform compared to expected results?
    The Random Forest regressor performed incredibly well for the English dataset, leveraging the text vectorization and sentiment analyses to create a near-100% accuracy, which was better than the previous Logistic Regression I tried.

For Spanish dataset from jpposadas:

- How well does TF-IDF work for feature extraction?
    The TF-IDF also worked very well for feature extraction. Spanish is a supported language for stop words and works similarly to english in TF-IDF methods. Next time, I might consider including accent removal in my preprocessing, however.
- What challenges did you encounter in text preprocessing?
    Again, preprocessing did indeed consuming a very large amount of computational resources, which was one of the most difficult challenges during this project. Next time, I might explore ways to cache preprocessed text.
- How did the Random Forest model perform compared to expected results?
    The Random Forest regressor performed decently for the Spanish dataset, leveraging the text vectorization and sentiment analyses to create a 80% accuracy. I am not sure if this was primarily due to the text feature transformation methods or model, but I would guess the Random Forest model was not the bottleneck for the performance of this model.

- Compare and contrast results from the primary dataset and the secondary dataset.
    The Kaggle English dataset provided was much more predictable (~100%) than the jpposadas Spanish dataset (~80%) with the TF-IDF Sentiment. This could be for multiple reasons, including that the Kaggle dataset might be designed for practice, that the Spanish dataset might have sourced data in a less consistent manner (as it was a personal dataset, it might have compiled varying samples in a less reliable fashion), or differences in culture in Spanish-speaking countries for fake news articles. Regardless, I found both of these results impressive: We can seriously just predict if articles are fake or not to a high degree of accuracy by counting instances of words they use!?