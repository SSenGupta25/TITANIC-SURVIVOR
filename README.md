# TITANIC-SURVIVOR

## Unveiling Survival Patterns on the Titanic: A Machine Learning Exploration (1898 characters)

This project delves into the tragic story of the Titanic disaster through the lens of machine learning. By leveraging historical passenger data, we aim to construct and evaluate predictive models capable of discerning survival probabilities during the shipwreck.

**Data Acquisition and Preprocessing:**

The cornerstone of this exploration is the Titanic passenger dataset, loaded and manipulated using the pandas library (`pd`). This data encompasses various passenger attributes believed to potentially influence survival outcomes, such as age, gender, social class, and fare paid.

**Feature Engineering:**

Data exploration and visualization, facilitated by libraries like matplotlib.pyplot (`plt`), play a crucial role in understanding the characteristics of the dataset. Feature engineering techniques might involve employing `LabelEncoder` to transform categorical variables into numerical representations suitable for machine learning algorithms. Additionally, dimensionality reduction techniques like Principal Component Analysis (`PCA`) or scaling techniques like `StandardScaler` from the scikit-learn library (`sklearn`) might be explored to optimize model performance.

**Model Selection and Training:**

A comprehensive array of machine learning algorithms from scikit-learn is considered for this investigation. These include:

* **Classification Algorithms:**
    * Multi-Layer Perceptron (MLP) Classifier (`MLPClassifier`) for non-linear relationships between features and survival.
    * K-Nearest Neighbors (KNN) Classifier (`KNeighborsClassifier`) to identify passengers with similar characteristics in the dataset.
    * Support Vector Machine (SVM) Classifier (`SVC`) for robust classification with high dimensionality.
    * Gaussian Process Classifier (`GaussianProcessClassifier`) with an RBF kernel for complex, non-linear modeling.
    * Decision Tree Classifier (`DecisionTreeClassifier`) for interpretable decision-making processes.
    * Ensemble Classifiers:
        * Random Forest Classifier (`RandomForestClassifier`) to leverage the power of multiple decision trees and enhance generalization.
        * AdaBoost Classifier (`AdaBoostClassifier`) for a stage-wise boosting approach that iteratively improves model performance.
    * Naive Bayes Classifier (`GaussianNB`) for efficient classification based on assumptions of feature independence.
    * Quadratic Discriminant Analysis (`QuadraticDiscriminantAnalysis`) for situations where the class distributions are assumed to be Gaussian.
* **Linear Classifier:** Logistic Regression (`LogisticRegression`) is employed for modeling the probability of survival based on the input features.

**Model Evaluation and Cross-Validation:**

To assess the effectiveness of each model, the dataset is split into training and testing sets using `train_test_split`. The training set is used to train the models, while the unseen testing set evaluates their generalizability and ability to predict survival outcomes for new data points. Cross-validation techniques implemented through `cross_validate` further enhance the robustness of the evaluation process.

**Conclusion:**

This project fosters a comprehensive understanding of machine learning methodologies by applying them to a real-world historical event. By constructing and evaluating various classification models, we gain valuable insights into the factors that might have influenced passenger survival on the Titanic. The project not only offers a historical perspective but also serves as a stepping stone for further exploration of advanced machine learning techniques. 

**Note:**

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
