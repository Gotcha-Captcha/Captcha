import mlflow
import matplotlib.pyplot as plt
import os

def generate_plots_from_mlflow():
    # Set experiment
    mlflow.set_experiment("Captcha_CRNN_CTC")
    
    # Get the latest run
    runs = mlflow.search_runs(experiment_names=["Captcha_CRNN_CTC"], order_by=["start_time DESC"])
    if runs.empty:
        print("No runs found.")
        return
    
    latest_run_id = runs.iloc[0].run_id
    print(f"Generating plots for run: {latest_run_id}")
    
    client = mlflow.tracking.MlflowClient()
    
    # Fetch metrics
    train_loss = [m.value for m in client.get_metric_history(latest_run_id, "train_loss")]
    val_loss = [m.value for m in client.get_metric_history(latest_run_id, "val_loss")]
    val_word_acc = [m.value for m in client.get_metric_history(latest_run_id, "val_word_accuracy")]
    val_char_acc = [m.value for m in client.get_metric_history(latest_run_id, "val_char_accuracy")]
    
    os.makedirs("docs/visualizations", exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # 1. Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('CTC Loss over Epochs (200 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 2. Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(val_word_acc, label='Word Accuracy (Exact)', color='blue')
    plt.plot(val_char_acc, label='Char Accuracy (Naive)', color='green', linestyle='--')
    plt.title('Accuracy over Epochs (200 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("docs/visualizations/training_history.png")
    plt.close()
    print("Plot saved to docs/visualizations/training_history.png")

if __name__ == "__main__":
    generate_plots_from_mlflow()
