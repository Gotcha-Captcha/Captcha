import mlflow
import json
import os
from pathlib import Path

def export_latest_metrics():
    # Set experiment
    mlflow.set_experiment("Captcha_CRNN_CTC")
    
    # Get the latest successful run
    runs = mlflow.search_runs(experiment_names=["Captcha_CRNN_CTC"], order_by=["start_time DESC"])
    if runs.empty:
        print("No MLflow runs found. Using default metadata.")
        return
    
    latest_run = runs.iloc[0]
    run_id = latest_run.run_id
    
    client = mlflow.tracking.MlflowClient()
    
    # Get metrics (use the last logged value)
    try:
        word_acc = client.get_metric_history(run_id, "val_word_accuracy")[-1].value
        char_acc = client.get_metric_history(run_id, "val_char_accuracy")[-1].value
        
        # New metrics with fallbacks
        try:
            precision = client.get_metric_history(run_id, "val_precision")[-1].value
            recall = client.get_metric_history(run_id, "val_recall")[-1].value
            f1_score = client.get_metric_history(run_id, "val_f1_score")[-1].value
            loss_value = client.get_metric_history(run_id, "val_loss")[-1].value
        except:
            precision, recall, f1_score, loss_value = 0.0, 0.0, 0.0, 0.0
        
        metadata = {
            "accuracy": f"{word_acc:.2%} (Word)",
            "char_accuracy": f"{char_acc:.2%}",
            "precision": f"{precision:.2f}",
            "recall": f"{recall:.2f}",
            "f1_score": f"{f1_score:.2f}",
            "loss_value": f"{loss_value:.4f}",
            "type": "CRNN (CNN + BLSTM)",
            "loss": "CTC Loss",
            "features": "Sequential Features",
            "preprocessing": "Resize (50x200) + Norm",
            "run_id": run_id,
            "timestamp": latest_run.start_time.isoformat() if hasattr(latest_run.start_time, 'isoformat') else str(latest_run.start_time)
        }
        
        output_path = Path("models/model_metadata.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"✅ Exported latest metrics from run {run_id} to {output_path}")
        
    except Exception as e:
        print(f"❌ Error exporting metrics: {e}")

if __name__ == "__main__":
    export_latest_metrics()
