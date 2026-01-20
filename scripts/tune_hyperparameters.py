import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skimage.io import imread
from scipy.stats import loguniform
import sys
import os

# Import our logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.model_logic import preprocess_image_v3, segment_characters_v2, extract_hog_features

def tune():
    import glob
    # SEARCH for the dataset directory
    paths = glob.glob("/Users/parkyoungdu/.cache/kagglehub/datasets/fournierp/captcha-version-2-images/versions/*/samples")
    if paths:
        images_dir = Path(paths[0])
        print(f"Found dataset at: {images_dir}")
    else:
        images_dir = Path(__file__).parent.parent / "dataset"
        print(f"Dataset not found in Kaggle cache, checking project: {images_dir}")
    
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    X = []
    y = []
    
    print(f"Loading {len(image_files)} images...")
    for path in image_files:
        try:
            img = imread(str(path))
            img_cleaned = preprocess_image_v3(img)
            char_images = segment_characters_v2(img_cleaned, num_chars=5)
            label_text = path.stem
            
            if len(char_images) == len(label_text):
                for char_img, char_label in zip(char_images, label_text):
                    feat = extract_hog_features(char_img)
                    X.append(feat)
                    y.append(char_label)
        except Exception as e:
            # print(f"Error processing {path}: {e}")
            continue
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Extracted features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Split TRAIN / TEST
    # We do NOT scale here. Scaling must be part of the pipeline to avoid leakage.
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    
    # Define Pipeline
    # scaling -> SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(class_weight='balanced')) # Added class_weight balanced just in case
    ])
    
    # Define Parameter Distribution for RandomizedSearch
    param_dist = {
        'svm__C': loguniform(1e-2, 1e2), # Log uniform distribution 0.01 to 100
        'svm__gamma': ['scale', 'auto'], # Removed fixed floats to let it decide or could add loguniform(1e-4, 1e-1)
        'svm__kernel': ['rbf'] # Strictly RBF for now as it performed best
    }
    
    print("Starting Randomized Search...")
    # n_iter=20 is a good balance between speed and coverage
    search = RandomizedSearchCV(
        pipeline, 
        param_dist, 
        n_iter=20, 
        cv=3, 
        verbose=2, 
        n_jobs=-1, 
        random_state=42,
        refit=True
    )
    
    search.fit(X_train, y_train)
    
    print(f"Best Parameters: {search.best_params_}")
    print(f"Best CV Score: {search.best_score_}")
    
    # Evaluate on Test Set
    # The pipeline (best_estimator_) handles scaling automatically for X_test
    best_model = search.best_estimator_
    test_accuracy = best_model.score(X_test, y_test)
    print(f"Test Set Accuracy: {test_accuracy}")
    
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    
    y_pred = best_model.predict(X_test)
    report = classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred))
    print("\nClassification Report:\n")
    print(report)
    
    # Save to MLflow
    try:
        import mlflow
        mlflow.set_experiment("Captcha_SVM_Tuning")
        with mlflow.start_run():
            # Log params - note they will be prefixed with svm__
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_score", search.best_score_)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_param("hog_pixels_per_cell", "(4, 4)")
            mlflow.log_param("search_strategy", "RandomizedSearchCV")
            
            # Create and log Confusion Matrix
            cm = confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred))
            plt.figure(figsize=(12, 10))
            if sns:
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
            else:
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.colorbar()
                tick_marks = np.arange(len(le.classes_))
                plt.xticks(tick_marks, le.classes_, rotation=45)
                plt.yticks(tick_marks, le.classes_)
                
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            
            # Log exact text report
            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt")
            
            # Save the actual model artifact just in case we want to grab it directly
            # logging the sklearn model
            mlflow.sklearn.log_model(best_model, "svm_model")
            
            print("Successfully logged metrics, parameters, and confusion matrix to MLflow.")
            
            # Cleanup
            if os.path.exists(cm_path): os.remove(cm_path)
            if os.path.exists("classification_report.txt"): os.remove("classification_report.txt")
            
    except ImportError:
        print("MLflow or Seaborn not installed, skipping logging.")
    except Exception as e:
        print(f"MLflow logging error: {e}")
    
if __name__ == "__main__":
    tune()
