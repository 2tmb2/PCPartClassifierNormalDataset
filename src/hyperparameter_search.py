"""
Hyperparameter search utility for PC parts classifier.
Tests multiple configurations and compares results.
"""
import itertools
import json
from pathlib import Path
from datetime import datetime
import torch
from train_pc_parts import train as original_train
import train_pc_parts

def run_hyperparameter_search(configs):
    """
    Run training with multiple hyperparameter combinations.
    
    Args:
        configs: Dict with lists of values for each hyperparameter
        Example:
        {
            'LEARNING_RATE': [1e-4, 5e-4, 1e-3],
            'BATCH_SIZE': [64, 128, 256],
            'OPTIMIZER': ['SGD', 'Adam', 'AdamW'],
        }
    """
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
            original_train()
            
            # Note: The training function saves the best model and prints results
            # We'll capture the best accuracy from the most recent checkpoint
            best_acc = get_latest_model_accuracy(train_pc_parts.OUTPUT_DIR)
            
            result = {
                'combination_id': combo_idx,
                'hyperparameters': combo_dict,
                'best_accuracy': best_acc,
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


def get_latest_model_accuracy(model_dir):
    """Extract accuracy from the most recent model checkpoint filename."""
    model_files = list(model_dir.glob("valacc*.pth"))
    if not model_files:
        return None
    
    # Get the most recent file
    latest = max(model_files, key=lambda p: p.stat().st_mtime)
    
    # Extract accuracy from filename: valacc{accuracy}pct_...
    filename = latest.stem
    try:
        acc_str = filename.split('valacc')[1].split('pct')[0]
        accuracy = float(acc_str) / 100
        return accuracy
    except (IndexError, ValueError):
        return None


if __name__ == "__main__":
    # Quick configuration search
    quick_configs = {
        'LEARNING_RATE': [1e-6, 1e-4, 1e-5],
        'BATCH_SIZE': [32, 64, 128, 256],
        'EPOCHS': [100, 300, 500],
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
