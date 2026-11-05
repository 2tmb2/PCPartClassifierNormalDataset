"""
Hyperparameter search utility for PC parts classifier.
Tests multiple configurations and compares results.
"""
import itertools
import json
from datetime import datetime
import torch
from train_pc_parts import train as original_train
import train_pc_parts

def run_hyperparameter_search(configs, *, early_stopping_patience=30):
    """
    Run training with multiple hyperparameter combinations.
    
    Args:
        configs: Dict with lists of values for each hyperparameter
        Example:
        {
            'LEARNING_RATE': [1e-4, 5e-4, 1e-3],
            'BATCH_SIZE': [64, 128, 256],
            'EPOCHS': [500],
        }
        early_stopping_patience: Stop a trial after this many epochs without
            lower validation loss. Use None to disable early stopping.
    """
    if early_stopping_patience is not None and (
        isinstance(early_stopping_patience, bool)
        or not isinstance(early_stopping_patience, int)
        or early_stopping_patience <= 0
    ):
        raise ValueError("Early stopping patience must be a positive integer or None.")
    results = []
    results_dir = train_pc_parts.OUTPUT_DIR / "hyperparameter_search_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    param_names = list(configs.keys())
    param_values = list(configs.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"Testing {len(combinations)} hyperparameter combinations")
    print("=" * 80)
    
    for combo_idx, combo_values in enumerate(combinations, 1):
        combo_dict = dict(zip(param_names, combo_values))
        
        print(f"\n[{combo_idx}/{len(combinations)}] Testing: {combo_dict}")
        print("-" * 80)
        
        # Save current values
        original_values = {}
        for param, value in combo_dict.items():
            original_values[param] = getattr(train_pc_parts, param)
            setattr(train_pc_parts, param, value)
        
        try:
            # Run training
            training_result = original_train(
                early_stopping_patience=early_stopping_patience
            )
            best_acc = training_result["best_accuracy"]
            
            result = {
                'combination_id': combo_idx,
                'hyperparameters': combo_dict,
                **training_result,
                'early_stopping_patience': early_stopping_patience,
                'timestamp': datetime.now().isoformat(),
            }
            results.append(result)
            
            print(f"✓ Best accuracy: {best_acc:.2%}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            result = {
                'combination_id': combo_idx,
                'hyperparameters': combo_dict,
                'best_accuracy': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }
            results.append(result)
        
        finally:
            # Restore original values
            for param, value in original_values.items():
                setattr(train_pc_parts, param, value)
    
    # Save and display results
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    
    # Sort by accuracy
    valid_results = [r for r in results if r['best_accuracy'] is not None]
    valid_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
    
    for rank, result in enumerate(valid_results[:10], 1):
        hp = result['hyperparameters']
        acc = result['best_accuracy']
        hp_str = ", ".join([f"{k}={v}" for k, v in hp.items()])
        print(f"{rank:2d}. {acc:.2%} | {hp_str}")
    
    # Save to file
    results_file = results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(valid_results[:10], f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return valid_results


if __name__ == "__main__":
    # Quick configuration search
    quick_configs = {
        'LEARNING_RATE': [1e-3, 1e-4, 1e-5],
        'BATCH_SIZE': [32, 64],
        'EPOCHS': [500],
    }
    
    print("Starting hyperparameter search...")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    
    results = run_hyperparameter_search(quick_configs)
    
    print("\n✓ Hyperparameter search complete!")
    if results:
        best = results[0]
        print(f"\nBest configuration:")
        for k, v in best['hyperparameters'].items():
            print(f"  {k}: {v}")
        print(f"  Accuracy: {best['best_accuracy']:.2%}")
