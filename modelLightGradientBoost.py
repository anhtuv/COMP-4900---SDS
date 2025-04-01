import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

def load_data(filepath):
    """ Load and preprocess dataset """
    data = pd.read_csv(filepath)
    
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Preprocess quantitative data (scaling)
    quantitative_data  = x.columns[x.dtypes == "int64"]
    scaler = StandardScaler()
    quantitative_preprocessed = pd.DataFrame(scaler.fit_transform(x[quantitative_data]), columns=quantitative_data)

    # Preprocess qualitative data (one-hot encoding)
    qualitative_data = x.columns[x.dtypes == "object"].values
    encoder = OneHotEncoder(sparse_output=False)
    qualitative_preprocessed = pd.DataFrame(
        encoder.fit_transform(x[qualitative_data]), 
        columns=encoder.get_feature_names_out(qualitative_data)
    )

    # Concatenate the preprocessed quantitative and qualitative data
    x_preprocessed = pd.concat([quantitative_preprocessed, qualitative_preprocessed], axis=1)

    # Map target variable ("Yes" and "No" to 1 and 0)
    y = y.map({"No": 0, "Yes": 1})
    
    return x_preprocessed, y

def train_model(x_train, y_train, x_val, y_val, x_test, y_test, x_train_full, y_train_full):
    """ Train the LightGBM model, evaluate performance, and plot metrics """
    model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val), (x_train, y_train)], eval_metric='logloss')

    # Predictions
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    # Accuracy scores
    train_acc = model.score(x_train, y_train)
    val_acc = model.score(x_val, y_val)
    test_acc = model.score(x_test, y_test)

    # F1 Scores
    train_f1 = metrics.f1_score(y_train, y_train_pred)
    val_f1 = metrics.f1_score(y_val, y_val_pred)
    test_f1 = metrics.f1_score(y_test, y_test_pred)

    # Log Loss (Training, Validation, and Testing)
    train_loss = metrics.log_loss(y_train, model.predict_proba(x_train)[:, 1])
    val_loss = metrics.log_loss(y_val, model.predict_proba(x_val)[:, 1])
    test_loss = metrics.log_loss(y_test, model.predict_proba(x_test)[:, 1])

    # Confusion matrix
    cm = metrics.confusion_matrix(y_test, y_test_pred)
    disp = metrics.ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])
    disp.plot(cmap="Blues_r")
    plt.savefig('confusion_matrix.png')

    # AUC Score
    auc = metrics.roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

    # Print the evaluation metrics in the requested format
    print(f"Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Loss: {test_loss:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("AUC Score:", auc)

    # Plot feature importance and metrics
    lgb.plot_importance(model)
    plt.savefig('importance.png')
    lgb.plot_metric(model)
    plt.savefig('metric.png')

def main():
    filepath = "Thyroid_Diff.csv"
    
    # Load and preprocess the data
    x_preprocessed, y = load_data(filepath)

    # Split the dataset into 70% training, 20% validation, and 10% testing
    x_train, x_temp, y_train, y_temp = train_test_split(x_preprocessed, y, test_size=0.30, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.33, random_state=42)

    # Train the LightGBM model and evaluate it
    train_model(x_train, y_train, x_val, y_val, x_test, y_test, x_train, y_train)

if __name__ == "__main__":
    main()
