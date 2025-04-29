# Breast Cancer Prediction using Logistic Regression

This project aims to predict whether a tumor is malignant (M) or benign (B) using a logistic regression model. The dataset contains features extracted from digitized images of breast cancer biopsies, and the goal is to classify the tumors as either malignant or benign.

## Libraries Used
- `numpy`: For numerical computations.
- `pandas`: For data manipulation and analysis.
- `matplotlib`: For plotting graphs.
- `scikit-learn`: For machine learning algorithms, model evaluation, and data preprocessing.

## Dataset
The dataset used in this project is the breast cancer dataset, which is available from the UCI Machine Learning Repository. It contains various features derived from cell nuclei present in breast cancer biopsies. The target variable is `diagnosis`, which contains two classes:
- **M**: Malignant (represented as 1)
- **B**: Benign (represented as 0)

The dataset is pre-processed by dropping unnecessary columns and encoding the target variable for classification.

## Workflow
### 1. Data Preprocessing
- **Load Dataset**: The dataset is loaded from a CSV file.
- **Drop Unnecessary Columns**: The columns `id` and `Unnamed: 32` are dropped since they are not relevant for prediction.
- **Target Encoding**: The target variable `diagnosis` is mapped from categorical values (`M`, `B`) to binary values (1 for malignant, 0 for benign).
- **Missing Values**: We check for any missing values in the dataset and handle them accordingly.

### 2. Feature Selection
- The features used for the prediction are all columns except the `diagnosis` column.

### 3. Train-Test Split
- The data is split into training and testing sets with an 80-20 ratio, ensuring that the model is trained on one set of data and tested on a separate set to evaluate performance.

### 4. Feature Scaling
- **Standardization**: The features are standardized using `StandardScaler` to bring them onto a common scale.

### 5. Model Training
- A **Logistic Regression** model is trained on the scaled training data with a maximum iteration of 1000.

### 6. Evaluation
- **Confusion Matrix**: We compute and display the confusion matrix for model performance.
- **Classification Report**: A detailed classification report is generated with precision, recall, f1-score, and support.
- **ROC-AUC Score**: The model's ability to distinguish between malignant and benign tumors is measured by the ROC-AUC score.

### 7. ROC Curve
- The **Receiver Operating Characteristic (ROC) Curve** is plotted to visualize the tradeoff between true positive rate and false positive rate at various thresholds.

### 8. Threshold Tuning
- The prediction threshold is tuned to 0.4, and a new confusion matrix is generated at this threshold to evaluate the model's performance with a different decision boundary.

### 9. Sigmoid Function Plot
- The sigmoid function is visualized for different values of `z` to show the relationship between the input and the predicted probability.

## Results
- The model performance is evaluated using the confusion matrix, classification report, and ROC curve.
- Threshold tuning allows you to adjust the decision boundary to improve specific metrics (e.g., precision or recall).
- The ROC curve and AUC score provide a visual and quantitative measure of the model's ability to classify cancerous tumors.

## How to Run
1. Clone or download the repository.
2. Place the `data.csv` dataset in the same directory as the script.
3. Run the script using Python 3.x:
   ```bash
   python breast_cancer_prediction.py
