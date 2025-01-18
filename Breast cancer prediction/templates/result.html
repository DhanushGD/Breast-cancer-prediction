import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import os

# Initialize Flask app
app = Flask(__name__)

# Load your dataset (assuming "data.csv" is in the correct location)
dataset = pd.read_csv("data.csv")

# Define features and target
prediction_features = ["radius_mean", 'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean',
                       'concave points_mean']
targeted_feature = 'diagnosis'

X = dataset[prediction_features]
y = dataset[targeted_feature]

# Label encode the target feature
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=15)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the classifiers
classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'ANN': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=15),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVC': SVC(kernel='linear', random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Voting Classifier (Ensemble Method)
ensemble_model = VotingClassifier(estimators=[
    ('KNN', classifiers['KNN']),
    ('ANN', classifiers['ANN']),
    ('Decision Tree', classifiers['Decision Tree'])
], voting='soft')


def generate_roc_curve(classifier, classifier_name):
    if hasattr(classifier, 'predict_proba'):
        fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
        plt.figure()
        plt.plot(fpr, tpr,
                 label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for {classifier_name}')
        plt.legend(loc='lower right')

        image_path = f'./static/roc_curve_{classifier_name}.png'
        plt.savefig(image_path)
        plt.close()
        return image_path
    return None


def train_and_evaluate(classifier_names):
    results = []
    for classifier_name in classifier_names:
        clf = classifiers.get(classifier_name, None)

        if clf:
            try:
                print(f"Training and evaluating {classifier_name}...")

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                if hasattr(clf, 'predict_proba'):
                    y_pred_proba = clf.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = clf.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                confusion = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred)

                auc = roc_auc_score(y_test, y_pred_proba) if hasattr(clf, 'predict_proba') else None

                # Generate ROC curve image
                roc_curve_path = generate_roc_curve(clf, classifier_name)

                prob = None
                sample_probabilities = None
                if hasattr(clf, 'predict_proba'):
                    prob = clf.predict_proba(X_test)
                    if prob is not None and len(prob) > 0:
                        sample_probabilities = {
                            "Malignant": prob[0][0],
                            "Benign": prob[0][1]
                        }

                result = {
                    'classifier': classifier_name,
                    'accuracy': accuracy,  # Now added 'accuracy' here.
                    'confusion_matrix': confusion,
                    'classification_report': report,
                    'roc_auc': auc,
                    'sample_probabilities': sample_probabilities,
                    'predictions': [("Malignant" if p == 0 else "Benign") for p in y_pred[:5]],
                    'roc_curve_image': roc_curve_path  # Add the ROC curve image path here
                }
                results.append(result)
            except Exception as e:
                print(f"Error occurred while evaluating {classifier_name}: {e}")
                results.append({
                    'classifier': classifier_name,
                    'accuracy': None,
                    'confusion_matrix': None,
                    'classification_report': None,
                    'roc_auc': None,
                    'sample_probabilities': None,
                    'predictions': [],
                    'roc_curve_image': None  # Ensure empty result for errors
                })

    return results


def comparative_analysis():
    accuracy_results = []
    for name, model in classifiers.items():
        try:
            print(f"Evaluating {name} for comparative analysis...")

            # Accuracy before hyperparameter tuning
            model.fit(X_train, y_train)
            accuracy_before = accuracy_score(y_test, model.predict(X_test))

            # Perform hyperparameter tuning and evaluate after
            best_params, best_score = hyperparameter_tuning(name)
            tuned_model = classifiers[name].set_params(**best_params)
            tuned_model.fit(X_train, y_train)
            accuracy_after = accuracy_score(y_test, tuned_model.predict(X_test))

            accuracy_percentage_before = f'{accuracy_before * 100:.2f}%'
            accuracy_percentage_after = f'{accuracy_after * 100:.2f}%'

            # Calculate AUC before and after tuning
            if hasattr(model, 'predict_proba'):
                auc_before = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            else:
                auc_before = None

            if hasattr(tuned_model, 'predict_proba'):
                auc_after = roc_auc_score(y_test, tuned_model.predict_proba(X_test)[:, 1])
            else:
                auc_after = None

            accuracy_results.append(
                {'classifier': name,
                 'accuracy_before': accuracy_before,
                 'accuracy_after': accuracy_after,
                 'auc_before': auc_before,
                 'auc_after': auc_after}
            )
        except Exception as e:
            print(f"Error occurred while evaluating {name}: {e}")
            accuracy_results.append({'classifier': name, 'accuracy_before': None, 'accuracy_after': None, 'auc_before': None, 'auc_after': None})

    accuracy_results = [result for result in accuracy_results if result['accuracy_before'] is not None]
    accuracy_results.sort(key=lambda x: x['accuracy_before'], reverse=True)

    return accuracy_results


def hyperparameter_tuning(classifier_name):
    clf = classifiers.get(classifier_name, None)
    if clf:
        param_grid = {
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'ANN': {
                'hidden_layer_sizes': [(100,), (50, 50), (100, 100), (150, 50)],
                'max_iter': [500, 1000, 1500],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'sgd', 'lbfgs']
            },
            'Decision Tree': {
                'max_depth': [3, 5, 7, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'SVC': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'degree': [3, 4, 5]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear', 'saga'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'max_iter': [100, 500, 1000]
            }
        }

        # Perform GridSearchCV with cross-validation
        grid_search = GridSearchCV(clf, param_grid[classifier_name], cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        return best_params, best_score
    return None, None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_classifiers = request.form.getlist('classifier')
        overview_options = request.form.getlist('overview')
        comparative_analysis_selected = 'Comparative Analysis' in request.form.getlist('comparative_analysis')

        overview_results = []
        if "Correlation Graph" in overview_options:
            overview_results.append({'type': 'Correlation Graph', 'image': '/static/correlation_graph.png'})

        if "Dataset Features" in overview_options:
            overview_results.append({'type': 'Dataset Features', 'image': '/static/dataset_features.png'})

        classifier_results = []
        if selected_classifiers:
            classifier_results = train_and_evaluate(selected_classifiers)

        comparative_results = None
        if comparative_analysis_selected:
            comparative_results = comparative_analysis()

        return render_template('result.html', overview_results=overview_results,
                               classifier_results=classifier_results, comparative_results=comparative_results)

    return render_template('index.html')


@app.route('/download_model', methods=['GET'])
def download_model():
    model = ensemble_model
    joblib.dump(model, 'static/ensemble_model.pkl')
    return send_file('static/ensemble_model.pkl', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
