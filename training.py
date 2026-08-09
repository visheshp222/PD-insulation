import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# Read file from user-provided Excel file
df = pd.read_excel("/content/graph_features_dataset.xlsx")

# Verify
print("\nFirst 5 Rows:\n")
display(df.head())

print("\nDataset Shape:", df.shape)



X = df.drop("Label", axis=1)
y = df["Label"]


encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nClasses:")
print(encoder.classes_)


# Check if there are enough samples and unique labels for train-test split
if len(X) < 2 or len(np.unique(y)) < 2:
    print("\nWarning: Not enough samples or unique labels for train-test split. Skipping model training.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    model_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

   

    model_rf.fit(X_train_scaled, y_train)

    print("\nRandom Forest Model Training Completed Successfully!")

  
    y_pred_rf = model_rf.predict(X_test_scaled)

   

    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    print("\nRandom Forest Accuracy = {:.2f}% ".format(accuracy_rf*100))

   

    print("\nRandom Forest Classification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred_rf,
            target_names=encoder.classes_
        )
    )

 

    cm_rf = confusion_matrix(y_test, y_pred_rf)

    print("\nRandom Forest Confusion Matrix:\n")
    print(cm_rf)

    disp_rf = ConfusionMatrixDisplay(
        confusion_matrix=cm_rf,
        display_labels=encoder.classes_
    )

    disp_rf.plot(cmap="Blues")
    plt.title("Random Forest Confusion Matrix")
    plt.show()


   

    importance_rf = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model_rf.feature_importances_
    })

    importance_rf = importance_rf.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nRandom Forest Feature Importance:\n")
    print(importance_rf)

    plt.figure(figsize=(10,6))
    plt.barh(
        importance_rf["Feature"],
        importance_rf["Importance"]
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()


    

    model_svc = SVC(random_state=42)

    

    model_svc.fit(X_train_scaled, y_train)

    print("\nSVM Model Training Completed Successfully!")

   
    y_pred_svc = model_svc.predict(X_test_scaled)

    

    accuracy_svc = accuracy_score(y_test, y_pred_svc)

    print("\nSVM Accuracy = {:.2f}% ".format(accuracy_svc*100))

    
    print("\nSVM Classification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred_svc,
            target_names=encoder.classes_
        )
    )

    

    cm_svc = confusion_matrix(y_test, y_pred_svc)

    print("\nSVM Confusion Matrix:\n")
    print(cm_svc)

    disp_svc = ConfusionMatrixDisplay(
        confusion_matrix=cm_svc,
        display_labels=encoder.classes_
    )

    disp_svc.plot(cmap="Blues")
    plt.title("SVM Confusion Matrix")
    plt.show()


   

    joblib.dump(model_rf, "rf_classifier.pkl")
    joblib.dump(model_svc, "svc_classifier.pkl")
    joblib.dump(encoder, "label_encoder.pkl")
    joblib.dump(scaler, "feature_scaler.pkl") # Save the scaler as well

    print("\nModels and Scaler Saved Successfully!")

   
    if not X_test.empty:
        sample = X_test.iloc[[0]]
        sample_scaled = scaler.transform(sample)

        prediction_rf = model_rf.predict(sample_scaled)
        probability_rf = model_rf.predict_proba(sample_scaled)
        label_rf = encoder.inverse_transform(prediction_rf)
        confidence_rf = probability_rf.max()*100

        print("\n===============================")
        print("Random Forest Prediction Result")
        print("===============================")
        print("Predicted Class :", label_rf[0])
        print("Confidence      : {:.2f}%",format(confidence_rf))
        print("===============================")
    else:
        print("\nNot enough samples in X_test to demonstrate single sample prediction.")