import json
from pathlib import Path
from ..core.config import STATE, MODELS_DIR

def load_metadata():
    # Load V1 Metadata
    v1_path = MODELS_DIR / "model_metadata_v1.json"
    if not v1_path.exists():
        v1_path = MODELS_DIR / "model_metadata.json" # Fallback
        
    if v1_path.exists():
        try:
            with open(v1_path, "r") as f:
                data = json.load(f)
                STATE["v1_metadata"].update(data)
                STATE["last_accuracy"] = data.get("accuracy", "0.00%")
                STATE["last_char_accuracy"] = data.get("char_accuracy", "0.00%")
                print(f"✅ V1 Metadata loaded from {v1_path.name}")
        except Exception as e:
            print(f"❌ Failed to load V1 metadata: {e}")

    # Load V2 Metadata
    v2_path = MODELS_DIR / "model_metadata_v2.json"
    if v2_path.exists():
        try:
            with open(v2_path, "r") as f:
                data = json.load(f)
                STATE["v2_metadata"].update(data)
                
                # Parse Stats for Dashboard
                if "fold_results" in data:
                    best_acc = 0.0
                    avg_loss = 0.0
                    total_folds = len(data["fold_results"])
                    
                    for fold in data["fold_results"]:
                        if fold["best_val_acc"] > best_acc:
                            best_acc = fold["best_val_acc"]
                        
                        # Take last validation loss as proxy
                        if "history" in fold and "val_loss" in fold["history"]:
                             avg_loss += fold["history"]["val_loss"][-1]
                    
                    if total_folds > 0:
                        avg_loss /= total_folds

                    STATE["v2_metadata"]["accuracy"] = f"{best_acc*100:.2f}%"
                    STATE["v2_metadata"]["loss"] = f"{avg_loss:.4f}"
                    STATE["v2_metadata"]["type"] = "EfficientNet-B0 (K-Fold)"
                
                print(f"✅ V2 Metadata loaded from {v2_path.name} (Acc: {STATE['v2_metadata']['accuracy']})")
        except Exception as e:
            print(f"❌ Failed to load V2 metadata: {e}")
    
    STATE["status"] = "Model metadata loaded"
    return STATE["v1_metadata"], STATE["v2_metadata"]
