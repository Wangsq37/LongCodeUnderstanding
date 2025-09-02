import os
import json
import re
import ast
from pathlib import Path
import pandas as pd
import argparse
from collections import defaultdict

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    # These will be set by command-line arguments
    PREDICTIONS_DIR_BASE = None
    EVALUATION_RESULTS_DIR = None
    
    # Static paths for ground truth data
    CALL_CHAIN_REPORTS_DIR = Path("output_results")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def load_jsonl(file_path: Path) -> list:
    data = []
    if not file_path.exists(): return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ WARNING: Skipping malformed JSON line in {file_path}")
    return data

def extract_file_list(raw_text: str) -> list[str]:
    """
    Robustly extracts a list of file paths from the model's raw string output.
    Handles JSON, Python list syntax, and other inconsistencies.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []
    
    # Prioritize finding the markdown block first
    code_block_match = re.search(r"```(?:output|json|python)?\s*(\[.*\])\s*```", raw_text, re.DOTALL)
    if code_block_match:
        list_str = code_block_match.group(1)
    else:
        # Fallback to finding any list-like structure
        list_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if not list_match:
            return []
        list_str = list_match.group(0)

    try:
        # Try parsing as JSON first, as it's the most strict format
        parsed_list = json.loads(list_str)
        if isinstance(parsed_list, list):
            return [str(item).strip() for item in parsed_list]
    except json.JSONDecodeError:
        try:
            # If JSON fails, try Python's literal_eval, which is safer than exec/eval
            parsed_list = ast.literal_eval(list_str)
            if isinstance(parsed_list, list):
                return [str(item).strip() for item in parsed_list]
        except (ValueError, SyntaxError, MemoryError, TypeError):
            # If all fails, return empty
            return []
    return []

def calculate_metrics(predicted_set: set, truth_set: set) -> dict:
    """Calculates Precision, Recall, F1-Score, and Exact Match."""
    if not truth_set and not predicted_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1}
    if not predicted_set: # Model predicted nothing
        return {"precision": 1.0 if not truth_set else 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 1 if not truth_set else 0}
    if not truth_set: # Ground truth was empty
        return {"precision": 0.0, "recall": 1.0 if not predicted_set else 0.0, "f1": 0.0, "exact_match": 1 if not predicted_set else 0}

    true_positives = len(predicted_set.intersection(truth_set))
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(truth_set)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1 if predicted_set == truth_set else 0
    
    return {"precision": precision, "recall": recall, "f1": f1, "exact_match": exact_match}

def evaluate_and_log_predictions(pred_file: Path, eval_file_handle, ground_truth_lookup: dict) -> list[dict]:
    """Processes a single prediction file, logs detailed results, and returns them for aggregation."""
    results = []
    reponame = pred_file.stem
    predictions = load_jsonl(pred_file)
    ratio_str = pred_file.parent.name

    for pred_item in predictions:
        test_file = pred_item.get("test_file")
        if not test_file or reponame not in ground_truth_lookup or test_file not in ground_truth_lookup[reponame]:
            continue

        truth_set = ground_truth_lookup[reponame][test_file]
        predicted_list = extract_file_list(pred_item.get("response", ""))
        predicted_set = set(predicted_list)
        
        metrics = calculate_metrics(predicted_set, truth_set)
        
        result_entry = {
            "reponame": reponame,
            "ratio": ratio_str,
            "task_id": pred_item.get("task_id"),
            "test_file": test_file,
            "sample_id": pred_item.get("sample_id"),
            "num_truth_files": len(truth_set),
            "num_pred_files": len(predicted_set),
            **metrics,
            "truth_files": sorted(list(truth_set)),
            "predicted_files": sorted(list(predicted_set))
        }
        eval_file_handle.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
        results.append(result_entry)
        
    return results

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate call chain prediction experiments and generate detailed reports.")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Base directory of the experiment predictions to evaluate.")
    args = parser.parse_args()

    Config.PREDICTIONS_DIR_BASE = Path(args.predictions_dir)
    if not Config.PREDICTIONS_DIR_BASE.exists():
        print(f"FATAL: Predictions directory not found at '{Config.PREDICTIONS_DIR_BASE}'")
        exit(1)
        
    base_name = Config.PREDICTIONS_DIR_BASE.name
    eval_name = base_name.replace("callchain_prediction_", "evaluation_results_")
    Config.EVALUATION_RESULTS_DIR = Path(eval_name)
    
    print("===== Starting Call Chain Prediction Evaluation =====")
    print(f"  - Input Predictions: {Config.PREDICTIONS_DIR_BASE}")
    print(f"  - Output Reports:    {Config.EVALUATION_RESULTS_DIR}")
    Config.EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- PASS 1: Pre-load all ground truth data ---
    print("\n--- PASS 1: Loading all ground truth call chains...")
    ground_truth_lookup = defaultdict(dict)
    repo_dirs = [d for d in Config.CALL_CHAIN_REPORTS_DIR.iterdir() if d.is_dir()]
    for repo_dir in repo_dirs:
        report_file = repo_dir / "report_files.jsonl"
        if report_file.exists():
            reponame = repo_dir.name
            for item in load_jsonl(report_file):
                test_file = item.get("test_file")
                # Exclude the test file itself from the ground truth set
                dependencies = {dep['file'] for dep in item.get('dependencies', []) if dep['file'] != test_file}
                ground_truth_lookup[reponame][test_file] = dependencies
    print(f"  -> Loaded ground truth for {len(ground_truth_lookup)} repositories.")

    # --- PASS 2: Find all prediction files and evaluate ---
    print("\n--- PASS 2: Evaluating all prediction files...")
    all_results = []
    predictions_base = Config.PREDICTIONS_DIR_BASE / "predictions"
    if not predictions_base.is_dir():
        print(f"FATAL: Subdirectory 'predictions' not found in '{Config.PREDICTIONS_DIR_BASE}'")
        return

    ratio_dirs = sorted([d for d in predictions_base.iterdir() if d.is_dir()])
    for ratio_dir in ratio_dirs:
        print(f"\n  Evaluating ratio: {ratio_dir.name}")
        eval_output_dir = Config.EVALUATION_RESULTS_DIR / ratio_dir.name
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        prediction_files = sorted(list(ratio_dir.glob("*.jsonl")))
        for pred_file in prediction_files:
            reponame = pred_file.stem
            eval_file_path = eval_output_dir / f"{reponame}.jsonl"
            with open(eval_file_path, 'w', encoding='utf-8') as f_out:
                repo_results = evaluate_and_log_predictions(pred_file, f_out, ground_truth_lookup)
                all_results.extend(repo_results)
            print(f"    - Evaluated '{reponame}', detailed results saved to '{eval_file_path}'")

    if not all_results:
        print("\nNo valid results found to generate a summary. Exiting.")
        return

    # --- PASS 3: Aggregate results and print summary tables ---
    print("\n\n--- PASS 3: Aggregating results for summary reports ---")
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 180)

    df_samples = pd.DataFrame(all_results)
    metric_cols = ['precision', 'recall', 'f1', 'exact_match']
    
    # Average metrics across samples for each unique task (defined by test_file and ratio)
    df_tasks = df_samples.groupby(['ratio', 'reponame', 'test_file'])[metric_cols].mean().reset_index()

    print("\n\n" + "="*80)
    print("  Overall Performance Summary (Macro-Averaged across Tasks)")
    print("="*80)
    
    # Macro-average across all tasks, grouped by ratio
    overall_summary = df_tasks.groupby('ratio')[metric_cols].mean()
    overall_summary['num_tasks'] = df_tasks.groupby('ratio').size()
    print(overall_summary.to_string(formatters={
        'precision': '{:,.4f}'.format, 'recall': '{:,.4f}'.format,
        'f1': '{:,.4f}'.format, 'exact_match': '{:,.4f}'.format
    }))

    print("\n\n" + "="*80)
    print("  Per-Repository Performance Summary (Macro-Averaged across Tasks)")
    print("="*80)

    # Macro-average across tasks, grouped by ratio and then repo
    repo_summary = df_tasks.groupby(['ratio', 'reponame'])[metric_cols].mean()
    repo_summary['num_tasks'] = df_tasks.groupby(['ratio', 'reponame']).size()
    print(repo_summary.to_string(formatters={
        'precision': '{:,.4f}'.format, 'recall': '{:,.4f}'.format,
        'f1': '{:,.4f}'.format, 'exact_match': '{:,.4f}'.format
    }))
    
    print("\n\n===== Evaluation Script Finished =====")

if __name__ == "__main__":
    main()