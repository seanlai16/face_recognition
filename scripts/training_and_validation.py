import time
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import data_preparation

# Hardcoded directory to model
model_dir = '../model'

# Name of model to train and evaluate
classifier_filename = os.path.join(model_dir,'classifier.pkl')

def main():
    start_time = time.time()

    # Load dataset
    labels, embeddings = data_preparation.load_dataset()

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split dataset into training & testing dataset
    x_train, x_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)

    # Define the SVM model
    model = SVC(random_state=42, probability=True)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['linear']
    }

    # Use GridSearchCV for hyperparameter tuning with k-fold cross-validation
    k = 5 
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_
    print(f'Best hyperparameters: {grid_search.best_params_}')
    print('Training Completed in {} seconds'.format(time.time() - start_time))

    # Save model
    with open(classifier_filename, 'wb') as outfile:
        pickle.dump((best_model, label_encoder), outfile)

    # Evaluate the best model using cross-validation
    cv_scores = cross_val_score(best_model, x_train, y_train, cv=kf)
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean cross-validation accuracy: {cv_scores.mean():.2f}')

    # Make predictions
    y_pred = best_model.predict(x_test)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    
    # Compute precision
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f'Precision: {precision}')

    # Compute recall (sensitivity)
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f'Recall (sensitivity): {recall}')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Compute specificity
    # Specificity = TN / (TN + FP)
    # For multi-class classification, compute specificity for each class
    specificity = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp))
    print(f'Specificity: {specificity}')


if __name__ == '__main__':
    main()