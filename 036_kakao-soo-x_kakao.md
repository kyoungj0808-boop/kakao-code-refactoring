원본코드
import os
import json
import yaml
import glob
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Set, List, Optional, Tuple
from loguru import logger
import traceback
import shutil
from src.utils.evaluation.evaluate_arguments import evaluate_sub_agent_history_f1
from src.utils.evaluation.evaluate_workflow_as_DAG import evaluate_workflow_multiple_runs
from src.utils.evaluation.eval_utils import (
    comprehensive_analysis,
    save_comprehensive_evaluation_results
)
from src.utils.model_factory import ModelFactory

def load_eval_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration from YAML file with environment variable overrides."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Apply environment variable overrides for judge config
        config = apply_judge_env_overrides(config)

        return config
    except FileNotFoundError:
        logger.error(f"Evaluation config file not found: {config_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading evaluation config: {e}")
        return {}


def apply_judge_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to judge config.

    Environment variables:
    - ORCHESTRATION_BENCH_JUDGE_PROVIDER: judge provider (openai, claude, gemini, bedrock)
    - ORCHESTRATION_BENCH_JUDGE_MODEL: judge model name
    - ORCHESTRATION_BENCH_JUDGE_BASE_URL: judge API base URL
    - ORCHESTRATION_BENCH_JUDGE_API_KEY: judge API key
    - ORCHESTRATION_BENCH_JUDGE_API_FORMAT: judge API format (openai, anthropic)
    - ORCHESTRATION_BENCH_JUDGE_TEMPERATURE: generation temperature
    - ORCHESTRATION_BENCH_JUDGE_MAX_TOKENS: max tokens
    - ORCHESTRATION_BENCH_JUDGE_TOP_P: top_p parameter
    - ORCHESTRATION_BENCH_JUDGE_AWS_ACCESS_KEY_ID: AWS access key for Bedrock
    - ORCHESTRATION_BENCH_JUDGE_AWS_SECRET_ACCESS_KEY: AWS secret key for Bedrock
    - ORCHESTRATION_BENCH_JUDGE_AWS_REGION: AWS region for Bedrock
    """
    judge_model = os.getenv("ORCHESTRATION_BENCH_JUDGE_MODEL")
    if not judge_model:
        return config

    # Initialize judge section if needed
    if "judge" not in config:
        config["judge"] = {"model": {}, "generation_params": {}}
    if "model" not in config["judge"]:
        config["judge"]["model"] = {}
    if "generation_params" not in config["judge"]:
        config["judge"]["generation_params"] = {}

    # Override judge model settings
    config["judge"]["model"]["model"] = judge_model

    provider = os.getenv("ORCHESTRATION_BENCH_JUDGE_PROVIDER")
    if provider:
        config["judge"]["model"]["provider"] = provider

    base_url = os.getenv("ORCHESTRATION_BENCH_JUDGE_BASE_URL")
    if base_url:
        config["judge"]["model"]["base_url"] = base_url

    api_key = os.getenv("ORCHESTRATION_BENCH_JUDGE_API_KEY")
    if api_key:
        config["judge"]["model"]["api_key"] = api_key

    api_format = os.getenv("ORCHESTRATION_BENCH_JUDGE_API_FORMAT")
    if api_format:
        config["judge"]["model"]["api_format"] = api_format

    # AWS Bedrock credentials
    aws_access_key_id = os.getenv("ORCHESTRATION_BENCH_JUDGE_AWS_ACCESS_KEY_ID")
    if aws_access_key_id:
        config["judge"]["model"]["AWS_ACCESS_KEY_ID"] = aws_access_key_id

    aws_secret_access_key = os.getenv("ORCHESTRATION_BENCH_JUDGE_AWS_SECRET_ACCESS_KEY")
    if aws_secret_access_key:
        config["judge"]["model"]["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    aws_region = os.getenv("ORCHESTRATION_BENCH_JUDGE_AWS_REGION")
    if aws_region:
        config["judge"]["model"]["AWS_REGION"] = aws_region

    # Override generation params
    temperature = os.getenv("ORCHESTRATION_BENCH_JUDGE_TEMPERATURE")
    if temperature:
        config["judge"]["generation_params"]["temperature"] = float(temperature)

    max_tokens = os.getenv("ORCHESTRATION_BENCH_JUDGE_MAX_TOKENS")
    if max_tokens:
        config["judge"]["generation_params"]["max_tokens"] = int(max_tokens)

    top_p = os.getenv("ORCHESTRATION_BENCH_JUDGE_TOP_P")
    if top_p:
        config["judge"]["generation_params"]["top_p"] = float(top_p)

    logger.info(f"Applied judge env overrides: model={judge_model}")

    return config

def load_evaluation_data(file_path: str) -> Dict[str, Any]:
    """Load evaluation data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"Evaluation data file not found: {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading evaluation data: {e}")
        return {}

async def evaluate_single_file(file_path: str, 
                               eval_config: Dict[str, Any], 
                               tool_descriptions: Dict[str, Any], 
                               save_llm_results: bool = False, 
                               llm_results_dir: str = None,
                               judge_model: ModelFactory = None) -> Dict[str, Any]:
    """Evaluate a single output file with both DAG and arguments evaluation."""
    logger.info(f"Evaluating file: {file_path}")
    agent_change_weight = eval_config.get("evaluation_params", {}).get("agent_change_weight", 0.8)
    status_change_weight = eval_config.get("evaluation_params", {}).get("status_change_weight", 0.5)
    # Load evaluation data
    data = load_evaluation_data(file_path)
    if not data:
        return {"error": f"Failed to load data from {file_path}"}
    
    try:
        # Workflow DAG evaluation (구조 및 상태 평가)
        logger.info("Running workflow DAG evaluation...")
        try:
            dag_results = evaluate_workflow_multiple_runs(data, agent_change_weight=agent_change_weight, status_change_weight=status_change_weight)
            if not isinstance(dag_results, dict):
                logger.warning(f"DAG evaluation returned unexpected type: {type(dag_results)}")
                dag_results = {
                    "total_score_with_failure": 0.0,
                    "total_score_without_failure": 0.0,
                    "average_structural_GED": 0.0,
                    "average_component_GED": 0.0,
                    "failed_workflow_generation": 0.0,
                    "total_comparison_steps_evaluated": 0.0,
                    "error": f"Unexpected return type: {type(dag_results)}"
                }
        except Exception as e:
            logger.warning(f"Workflow DAG evaluation failed: {e}")
            dag_results = {
                "total_score_with_failure": 0.0,
                "total_score_without_failure": 0.0,
                "average_structural_GED": 0.0,
                "average_component_GED": 0.0,
                "failed_workflow_generation": 0.0,
                "total_comparison_steps_evaluated": 0.0,
                "error": str(e)
            }
        
        # Arguments evaluation (도구 호출 평가)
        logger.info("Running arguments evaluation...")
        if eval_config:
            try:
                # Create file identifier for LLM results
                file_identifier = os.path.splitext(os.path.basename(file_path))[0] if save_llm_results else None

                args_results = await evaluate_sub_agent_history_f1(
                    data, 
                    eval_config,
                    tool_descriptions,
                    judge_model=judge_model,
                    save_llm_results=save_llm_results,
                    output_dir=llm_results_dir,
                    file_identifier=file_identifier
                )
                                
                # Ensure args_results is a dictionary
                if not isinstance(args_results, dict):
                    logger.warning(f"Arguments evaluation returned unexpected type: {type(args_results)}")

                    args_results = {
                        "key_score": {"avg_f1": 0.0}, 
                        "value_score_not_count_rejection": {"avg_f1": 0.0},
                        "function_name_score": {"f1-score": 0.0},
                        "total_rejection_cases": 0.0,
                        "total_true_positive_reject": 0.0,
                        "total_true_negative_reject": 0.0,
                        "total_false_positive_reject": 0.0,
                        "total_false_negative_reject": 0.0,
                        "total_true_positive_fc": 0.0,  # FC True Positive
                        "total_false_positive_fc": 0.0,  # FC False Positive  
                        "total_false_negative_fc": 0.0,  # FC False Negative
                        "total_true_negative_fc": 0.0,  # FC True Negative
                        "llm_failed_trials": [],
                        "total_function_call_cases": 0.0,
                        "error": f"Unexpected return type: {type(args_results)}"
                    }
                    
            except Exception as e:
                logger.warning(f"{traceback.print_exc()}")
                logger.warning(f"Arguments evaluation failed: {e}")
                args_results = {
                    "key_score": {"avg_f1": 0.0}, 
                    "value_score_not_count_rejection": {"avg_f1": 0.0},
                    "function_name_score": {"f1-score": 0.0},
                    "total_rejection_cases": 0.0,
                    "total_true_positive_reject": 0.0,
                    "total_true_negative_reject": 0.0,
                    "total_false_positive_reject": 0.0,
                    "total_false_negative_reject": 0.0,
                    "total_true_positive_fc": 0.0,  # FC True Positive
                    "total_false_positive_fc": 0.0,  # FC False Positive  
                    "total_false_negative_fc": 0.0,  # FC False Negative
                    "total_true_negative_fc": 0.0,  # FC True Negative
                    "llm_failed_trials": [],
                    "total_function_call_cases": 0.0,
                    "error": str(e)
                }
        else:
            logger.warning("No evaluation config provided, skipping arguments evaluation")
            args_results = {
                "key_score": {"avg_f1": 0.0}, 
                "value_score_not_count_rejection": {"avg_f1": 0.0},
                "function_name_score": {"f1-score": 0.0},
                "total_rejection_cases": 0.0,
                "total_true_positive_reject": 0.0,
                "total_true_negative_reject": 0.0,
                "total_false_positive_reject": 0.0,
                "total_false_negative_reject": 0.0,
                "total_true_positive_fc": 0.0,  # FC True Positive
                "total_false_positive_fc": 0.0,  # FC False Positive  
                "total_false_negative_fc": 0.0,  # FC False Negative
                "total_true_negative_fc": 0.0,  # FC True Negative
                "llm_failed_trials": [],
                "total_function_call_cases": 0.0,
                "error": "No evaluation config provided"
                }

        combined_results = {
            "file_path": file_path,
            "workflow_evaluation_with_failure": dag_results.get("total_score_with_failure", 0.0),
            "workflow_evaluation_without_failure": dag_results.get("total_score_without_failure", 0.0),
            "average_structural_GED": dag_results.get("average_structural_GED", 0.0),
            "average_component_GED": dag_results.get("average_component_GED", 0.0),
            "failed_workflow_generation": dag_results.get("failed_workflow_generation", 0),
            "total_comparison_steps_evaluated": dag_results.get("total_comparison_steps_evaluated", 0),
            "arguments_key_score": args_results.get("key_score", {}).get("avg_f1", 0.0) if isinstance(args_results, dict) else 0.0,
            "arguments_value_score_not_count_rejection": args_results.get("value_score_not_count_rejection", {}).get("avg_f1", 0.0) if isinstance(args_results, dict) else 0.0,
            "total_fc_successful_calls": args_results.get("total_fc_successful_calls", 0.0),
            # "total_perfect_function_calls": args_results.get("total_perfect_function_calls", 0.0),
            "total_rejection_cases": args_results.get("total_rejection_cases", 0.0),
            "total_true_positive_reject": args_results.get("total_true_positive_reject", 0.0),
            "total_true_negative_reject": args_results.get("total_true_negative_reject", 0.0),
            "total_false_positive_reject": args_results.get("total_false_positive_reject", 0.0),
            "total_false_negative_reject": args_results.get("total_false_negative_reject", 0.0),
            "total_true_positive_fc": args_results.get("total_true_positive_fc", 0.0),  # FC True Positive
            "total_false_positive_fc": args_results.get("total_false_positive_fc", 0.0),  # FC False Positive
            "total_false_negative_fc": args_results.get("total_false_negative_fc", 0.0),  # FC False Negative  
            "total_true_negative_fc": args_results.get("total_true_negative_fc", 0.0),  # FC True Negative
            "total_function_call_cases": args_results.get("total_function_call_cases", 0.0),
            "function_name_accuracy": args_results.get("function_name_score", {}).get("f1-score", 0.0) if isinstance(args_results, dict) else 0.0,
            "detail": {"workflow_detail": dag_results,
                      "arguments_detail": args_results
                    }
        }

        # Save comprehensive evaluation results if requested
        if save_llm_results and llm_results_dir and file_identifier:
            save_comprehensive_evaluation_results(combined_results, llm_results_dir, file_identifier)

        logger.success(f"✅ Successfully evaluated: {file_path}")
        logger.debug(f"Detailed Results: {combined_results}")
        return combined_results

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error evaluating {file_path}: {e}")
        return {"error": str(e), "file_path": file_path}

async def evaluate_directory(directory_path: str, eval_config: Dict[str, Any], 
                           tool_descriptions: Dict[str, Any],
                           pattern: str = "*.json", sequential: bool = False,
                           output_path: str = None, save_llm_results: bool = False,
                           llm_results_dir: str = None,
                           start_over: bool = False,
                           judge_model: ModelFactory = None) -> Dict[str, Any]:
    """Evaluate all JSON files in a directory with resume capability."""
    logger.info(f"Evaluating directory: {directory_path}")
    
    # Setup temp directory for incremental results
    temp_dir = os.path.join(os.path.dirname(output_path) if output_path else ".", ".temp_eval_results")
    if start_over:
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            os.remove(output_path) 
        except: pass
    # Find all JSON files matching the pattern
    search_pattern = os.path.join(directory_path, "**", pattern)
    json_files = glob.glob(search_pattern, recursive=True)
    
    if not json_files:
        logger.warning(f"No files found matching pattern: {search_pattern}")
        return {"error": "No files found", "pattern": search_pattern}
    
    logger.info(f"Found {len(json_files)} files to evaluate")
    
    # Check for existing results and determine what needs to be evaluated
    existing_results = {}
    completed_files = set()
    if output_path and os.path.exists(output_path):
        existing_results = load_existing_results(output_path)
        completed_files = get_completed_files(existing_results)
        logger.info(f"Found {len(completed_files)} already completed files")
    
    # Also check for incremental results
    incremental_results = load_incremental_results(temp_dir)
    for result in incremental_results:
        if "file_path" in result and "error" not in result:
            completed_files.add(result["file_path"])
    
    # Filter out already completed files
    remaining_files = [f for f in json_files if f not in completed_files]
    logger.info(f"Need to evaluate {len(remaining_files)} remaining files")
    
    # Start with existing results
    all_results = []
    if "detailed_results" in existing_results:
        all_results.extend(existing_results["detailed_results"])
    all_results.extend(incremental_results)
    
    if remaining_files:
        # Evaluate remaining files
        total_files = len(remaining_files)
        
        if sequential:
            logger.info(f"Starting sequential evaluation of {total_files} remaining files...")
            for i, file_path in enumerate(remaining_files, 1):
                logger.info(f"Processing file {i}/{total_files}: {os.path.basename(file_path)}")
                try:
                    result = await evaluate_single_file(file_path, 
                                                        eval_config, 
                                                        tool_descriptions, 
                                                        save_llm_results, 
                                                        llm_results_dir,
                                                        judge_model
                                                        )
                    all_results.append(result)
                    # Save incremental result
                    save_incremental_result(result, temp_dir, len(all_results))
                    logger.success(f"✅ Completed {i}/{total_files}: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Failed to evaluate {file_path}: {e}")
                    error_result = {"error": str(e), "file_path": file_path}
                    all_results.append(error_result)
                    save_incremental_result(error_result, temp_dir, len(all_results))
        else:
            logger.info(f"Starting concurrent evaluation of {total_files} remaining files...")
            
            # Create tasks for remaining files
            tasks = []
            for i, file_path in enumerate(remaining_files, 1):
                logger.info(f"Queuing file {i}/{total_files}: {os.path.basename(file_path)}")
                task = evaluate_single_file(file_path, eval_config, tool_descriptions, save_llm_results, llm_results_dir, judge_model)
                tasks.append(task)
            
            # Execute all tasks concurrently and collect results
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and handle exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"File evaluation failed for {os.path.basename(remaining_files[i])}: {result}")
                        error_result = {"error": str(result), "file_path": remaining_files[i]}
                        all_results.append(error_result)
                    else:
                        all_results.append(result)
                    # Save incremental result
                    save_incremental_result(all_results[-1], temp_dir, len(all_results))
                    
            except Exception as e:
                logger.error(f"Concurrent evaluation failed: {e}")
                # Fall back to sequential processing for remaining files
                logger.info("Falling back to sequential processing...")
                for i, file_path in enumerate(remaining_files, 1):
                    logger.info(f"Processing file {i}/{total_files}: {os.path.basename(file_path)}")
                    try:
                        result = await evaluate_single_file(file_path, eval_config, tool_descriptions, save_llm_results, llm_results_dir)
                        all_results.append(result)
                    except Exception as file_error:
                        logger.error(f"Failed to evaluate {file_path}: {file_error}")
                        all_results.append({"error": str(file_error), "file_path": file_path})
                    save_incremental_result(all_results[-1], temp_dir, len(all_results))
    
    # Calculate overall statistics
    successful_evals = [r for r in all_results if "error" not in r]
    failed_evals = [r for r in all_results if "error" in r]

    if successful_evals:
        avg_workflow_scores_with_failure = sum(r.get("workflow_evaluation_with_failure", 0) or 0 for r in successful_evals) / len(successful_evals)
        avg_workflow_scores_without_failure = sum(r.get("workflow_evaluation_without_failure", 0) or 0 for r in successful_evals) / len(successful_evals)
        avg_structural_GED = sum(r.get("average_structural_GED", 0) or 0 for r in successful_evals) / len(successful_evals)
        avg_component_GED = sum(r.get("average_component_GED", 0) or 0 for r in successful_evals) / len(successful_evals)
        avg_arguments_key_score = sum(r.get("arguments_key_score", 0) or 0 for r in successful_evals) / len(successful_evals)
        avg_arguments_value_score = sum(r.get("arguments_value_score_not_count_rejection", 0) or 0 for r in successful_evals) / len(successful_evals)
        total_rejection_cases = sum(r.get("total_rejection_cases", 0) or 0 for r in successful_evals) 
        total_true_positive_reject = sum(r.get("total_true_positive_reject", 0) or 0 for r in successful_evals) 
        total_true_negative_reject = sum(r.get("total_true_negative_reject", 0) or 0 for r in successful_evals) 
        total_false_positive_reject = sum(r.get("total_false_positive_reject", 0) or 0 for r in successful_evals) 
        total_false_negative_reject = sum(r.get("total_false_negative_reject", 0) or 0 for r in successful_evals) 
        total_true_positive_fc = sum(r.get("total_true_positive_fc", 0) or 0 for r in successful_evals)  # FC True Positive
        total_false_positive_fc = sum(r.get("total_false_positive_fc", 0) or 0 for r in successful_evals)  # FC False Positive
        total_false_negative_fc = sum(r.get("total_false_negative_fc", 0) or 0 for r in successful_evals)  # FC False Negative
        total_true_negative_fc = sum(r.get("total_true_negative_fc", 0) or 0 for r in successful_evals)  # FC True Negative
        total_rejection_type_mismatch = sum(r.get("total_rejection_type_mismatch", 0) or 0 for r in successful_evals)  # Rejection 타입 불일치
        total_failed_generation = sum(r.get("total_failed_generation", 0) or 0 for r in successful_evals)  # 예측 생성 실패
        total_function_call_cases = sum(r.get("total_function_call_cases", 0) or 0 for r in successful_evals) 
        avg_function_name_accuracy = sum(r.get("function_name_accuracy", 0) or 0 for r in successful_evals) / len(successful_evals)
        total_failed_workflow_generation = sum(r.get("failed_workflow_generation", 0) or 0 for r in successful_evals)
        total_fc_successful_calls = sum(r.get("total_fc_successful_calls", 0) or 0 for r in successful_evals)
        total_comparison_steps_evaluated = sum(r.get("total_comparison_steps_evaluated", 0) or 0 for r in successful_evals)
        
        # 전체 통계를 위한 confusion matrix 데이터 생성
        overall_confusion_matrix = {
            "total_true_positive_reject": total_true_positive_reject,
            "total_true_negative_reject": total_true_negative_reject,
            "total_false_positive_reject": total_false_positive_reject,
            "total_false_negative_reject": total_false_negative_reject,
            "total_true_positive_fc": total_true_positive_fc,  # FC True Positive
            "total_false_positive_fc": total_false_positive_fc,  # FC False Positive
            "total_false_negative_fc": total_false_negative_fc,  # FC False Negative
            "total_true_negative_fc": total_true_negative_fc,  # FC True Negative
            "total_rejection_cases": total_rejection_cases,
            "total_function_call_cases": total_function_call_cases
        }
        
        # 전체 데이터에 대한 comprehensive analysis 수행
        overall_analysis = comprehensive_analysis(overall_confusion_matrix)

        # Call/Rejection Classification Accuracy 계산
        # = (reject_f1 + fc_f1) / 2
        # reject_f1: 거부 결정 성능 F1
        # fc_f1: 함수 호출 결정 성능 F1
        reject_f1 = overall_analysis.get("reject_decision_performance", {}).get("f1", 0)
        fc_f1 = overall_analysis.get("fc_decision_performance", {}).get("f1", 0)
        call_rejection_acc = round((reject_f1 + fc_f1) / 2, 4)

        # FC Score 계산
        # = (function_name_accuracy + arguments_key_score + arguments_value_score) / 3
        # - function_name_accuracy: 함수 이름 F1 스코어
        # - arguments_key_score: 파라미터 키 F1 스코어
        # - arguments_value_score: 파라미터 값 F1 스코어 (rejection 케이스 제외)
        fc_acc = round((avg_function_name_accuracy + avg_arguments_key_score + avg_arguments_value_score) / 3, 4)
        plan_score = round(avg_workflow_scores_with_failure, 4)
        average_score = round((call_rejection_acc + fc_acc + plan_score) / 3, 4)
        
        overall_stats = {
            "total_files": len(json_files),
            "successful_evaluations": len(successful_evals),
            "failed_evaluations": len(failed_evals),
            "total_failed_workflow_generation": total_failed_workflow_generation,
            "total_comparison_steps_evaluated": total_comparison_steps_evaluated,
            "key_metrics": {
                "Average": average_score,
                "Call Rejection Classification Accuracy": call_rejection_acc,
                "FC": fc_acc,
                "Plan": plan_score
            },
            "score_details": {
                "workflow_scores": {
                    "workflow_evaluation_with_failure": round(avg_workflow_scores_with_failure, 4),
                    "workflow_evaluation_without_failure": round(avg_workflow_scores_without_failure, 4),
                    "average_structural_GED": round(avg_structural_GED, 4),
                    "average_component_GED": round(avg_component_GED, 4)
                },
                "function_call_scores": {
                    "function_name_accuracy": round(avg_function_name_accuracy, 4),
                    "arguments_key_score": round(avg_arguments_key_score, 4),
                    "arguments_value_score_not_count_rejection": round(avg_arguments_value_score, 4),
                    "overall_function_call_failed_rate": round((total_function_call_cases - total_fc_successful_calls)/total_function_call_cases, 4) if total_function_call_cases > 0 else 0.0,
                    "total_fc_successful_calls": total_fc_successful_calls,
                    "total_function_call_cases": total_function_call_cases
                },
                "call_rejection_scores": {
                    "total_rejection_cases": total_rejection_cases,
                    "reject_decision_performance": overall_analysis.get("reject_decision_performance", {}),
                    "fc_decision_performance": overall_analysis.get("fc_decision_performance", {}),
                    "error_patterns": overall_analysis.get("error_patterns", {}),
                    "rejection_type_accuracy": round((total_true_positive_reject - total_rejection_type_mismatch) / total_true_positive_reject, 4) if total_true_positive_reject > 0 else 0.0,
                    "total_rejection_type_mismatch": total_rejection_type_mismatch,
                    "total_failed_generation": total_failed_generation
                },
                "raw_confusion_matrix": {
                    "total_true_positive_reject": total_true_positive_reject,
                    "total_true_negative_reject": total_true_negative_reject,
                    "total_false_positive_reject": total_false_positive_reject,
                    "total_false_negative_reject": total_false_negative_reject,
                    "total_true_positive_fc": total_true_positive_fc,
                    "total_false_positive_fc": total_false_positive_fc,
                    "total_false_negative_fc": total_false_negative_fc,
                    "total_true_negative_fc": total_true_negative_fc
                }
            }
        }
    else:
        overall_stats = {
            "total_files": len(json_files),
            "successful_evaluations": 0,
            "failed_evaluations": len(failed_evals),
            "total_failed_workflow_generation": 0,
            "total_comparison_steps_evaluated": 0,
            "key_metrics": {
                "Average": 0.0,
                "Call Rejection Classification Accuracy": 0.0,
                "FC": 0.0,
                "Plan": 0.0
            },
            "score_details": {
                "workflow_scores": {
                    "workflow_evaluation_with_failure": 0.0,
                    "workflow_evaluation_without_failure": 0.0,
                    "average_structural_GED": 0.0,
                    "average_component_GED": 0.0
                },
                "function_call_scores": {
                    "function_name_accuracy": 0.0,
                    "arguments_key_score": 0.0,
                    "arguments_value_score_not_count_rejection": 0.0,
                    "overall_function_call_failed_rate": 0.0,
                    "total_fc_successful_calls": 0.0,
                    "total_function_call_cases": 0.0
                },
                "call_rejection_scores": {
                    "total_rejection_cases": 0.0,
                    "reject_decision_performance": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "fc_decision_performance": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0},
                    "error_patterns": {"overaction_rate": 0.0, "underaction_rate": 0.0}
                },
                "raw_confusion_matrix": {
                    "total_true_positive_reject": 0.0,
                    "total_true_negative_reject": 0.0,
                    "total_false_positive_reject": 0.0,
                    "total_false_negative_reject": 0.0,
                    "total_true_positive_fc": 0.0,
                    "total_false_positive_fc": 0.0,
                    "total_false_negative_fc": 0.0,
                    "total_true_negative_fc": 0.0
                }
            }
        }
    
    final_results = {
        "directory": directory_path,
        "overall_statistics": overall_stats,
        "detailed_results": all_results
    }
    # Save final results and cleanup
    if output_path:
        save_results(final_results, output_path)
        cleanup_temp_dir(temp_dir)
        
    return final_results

def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    try:
        # Get directory path, handle case where output_path is just a filename
        dir_path = os.path.dirname(output_path)
        if dir_path:  # Only create directory if there's a path component
            os.makedirs(dir_path, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.success(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def load_existing_results(output_path: str) -> Dict[str, Any]:
    """Load existing evaluation results if available."""
    try:
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded existing results from: {output_path}")
            return results
    except Exception as e:
        logger.warning(f"Could not load existing results: {e}")
    return {}

def get_completed_files(existing_results: Dict[str, Any]) -> Set[str]:
    """Extract list of already completed files from existing results."""
    completed_files = set()
    if "detailed_results" in existing_results:
        for result in existing_results["detailed_results"]:
            if "file_path" in result and "error" not in result:
                completed_files.add(result["file_path"])
    return completed_files

def save_incremental_result(result: Dict[str, Any], temp_dir: str, file_index: int):
    """Save individual file result for recovery purposes."""
    try:
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"result_{file_index:04d}.json")
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Could not save incremental result: {e}")

def load_incremental_results(temp_dir: str) -> List[Dict[str, Any]]:
    """Load all incremental results from temp directory."""
    results = []
    try:
        if os.path.exists(temp_dir):
            temp_files = sorted(glob.glob(os.path.join(temp_dir, "result_*.json")))
            for temp_file in temp_files:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    results.append(result)
            logger.info(f"Loaded {len(results)} incremental results from temp directory")
    except Exception as e:
        logger.warning(f"Could not load incremental results: {e}")
    return results

def cleanup_temp_dir(temp_dir: str):
    """Clean up temporary result files."""
    try:
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary result files")
    except Exception as e:
        logger.warning(f"Could not clean up temp directory: {e}")

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive workflow evaluation tool")
    parser.add_argument("--input", required=True,
                      help="Input file or directory path to evaluate")
    parser.add_argument("--agent-cards-path", default="data/EN/multiagent_cards",
                      help="Path to the directory containing agent card JSON files")
    parser.add_argument("--eval-config", default="config/base_config/eval_config.yaml",
                      help="Path to evaluation configuration file")
    parser.add_argument("--output", 
                      help="Output file path for results (optional)")
    parser.add_argument("--pattern", default="*_out.json",
                      help="File pattern to match when evaluating directory")
    parser.add_argument("--mode", choices=["file", "directory"], default="auto",
                      help="Evaluation mode: file or directory (auto-detect if not specified)")
    parser.add_argument("--max-concurrent", type=int, default=5,
                      help="Maximum number of concurrent LLM API calls")
    parser.add_argument("--batch-size", type=int, default=10,
                      help="Batch size for processing LLM evaluation calls")
    parser.add_argument("--skip-llm-eval", action="store_true",
                      help="Skip LLM-based argument evaluation (faster, only key-based evaluation)")
    parser.add_argument("--sequential", action="store_true",
                      help="Process files sequentially instead of concurrently (slower but more stable)")
    parser.add_argument("--save-llm-results", action="store_true",
                      help="Save LLM evaluation inputs and outputs for analysis")
    parser.add_argument("--llm-results-dir", default="llm_evaluation_logs",
                      help="Directory to save LLM evaluation results")
    parser.add_argument("--start-over", action="store_true",
                    help="Start over from scratch (delete temp directory)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                      default="INFO", help="Set logging level")

    args = parser.parse_args()
    
    # Configure logger level
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level=args.log_level)
    
    # Load evaluation configuration
    eval_config = load_eval_config(args.eval_config)
    
    # Add performance options to eval_config
    if eval_config:
        eval_config["max_concurrent"] = args.max_concurrent
        eval_config["batch_size"] = args.batch_size
        eval_config["skip_llm_eval"] = args.skip_llm_eval
    
    judge_model = None
    if not args.skip_llm_eval:
        judge_config = eval_config.get("judge", {}).get("model", {})
        if not judge_config.get("model"):
            logger.error("Judge model not specified in eval_config while LLM evaluation is enabled.")
            return
        judge_model = await ModelFactory.create_model(judge_config)
    # Determine evaluation mode
    if args.mode == "auto":
        if os.path.isfile(args.input):
            mode = "file"
        elif os.path.isdir(args.input):
            mode = "directory"
        else:
            logger.error(f"Input path does not exist: {args.input}")
            return
    else:
        mode = args.mode
    
    tool_descriptions = {}
    agent_card_pathes = glob.glob(os.path.join(args.agent_cards_path, "*.json"))
    for path in agent_card_pathes:
        with open(path, 'r', encoding='utf-8') as f:
            agent_card = json.load(f)
        tools = agent_card['tools']
        for tool in tools:
            tool_descriptions[tool['name']] = tool['parameters']

    # Run evaluation
    try:
        if mode == "file":
            logger.info(f"Evaluating single file: {args.input}")
            results = await evaluate_single_file(args.input, eval_config, tool_descriptions, args.save_llm_results, args.llm_results_dir)
            
            if "error" not in results:
                logger.info(f"result: {results}")
        elif mode == "directory":
            logger.info(f"Evaluating directory: {args.input}")
            # Generate default output path if not specified
            default_output = os.path.join(args.input, f"{os.path.basename(args.input)}-result.json")
            output_path = args.output if args.output else default_output
            if args.start_over:
                shutil.rmtree(output_path, ignore_errors=True)
            results = await evaluate_directory(args.input, 
                                                eval_config, 
                                                tool_descriptions,
                                                args.pattern, 
                                                args.sequential, 
                                                output_path, 
                                                args.save_llm_results, 
                                                args.llm_results_dir,
                                                args.start_over,
                                                judge_model)

            # Print summary
            if "error" not in results:
                stats = results["overall_statistics"]
                logger.info("📊 Overall Evaluation Summary:")
                logger.info(f"  Total Files: {stats['total_files']}")
                logger.info(f"  Successful: {stats['successful_evaluations']}")
                logger.info(f"  Failed: {stats['failed_evaluations']}")
                logger.info(f"  Total Failed Workflow Generation: {stats.get('total_failed_workflow_generation', 0)}")
                logger.info("  Average Metrics:")
                logger.info(f": {stats}")
        # Save results if output path specified (for single file mode)
        if mode == "file" and args.output:
            save_results(results, args.output)
        elif mode == "file" and not args.output:
            # Print JSON results for single file
            print(json.dumps(results, indent=2, ensure_ascii=False))
        # For directory mode, results are already saved in evaluate_directory function
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

평가 엔진의 복구성·예외 격리·메트릭 설계는 이미 엔터프라이즈급 수준이지만, 실제 운영 환경에서는 동시성 제어 없는 대량 비동기 호출과 Judge Model 생명주기 미관리라는 두 개의 런타임 폭탄이 남아 있어 9.5+ 프로덕션 등급 직전에서 멈춘 구조다.

제안패치
import os
import json
import yaml
import glob
import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Set, List, Optional, Tuple
from contextlib import asynccontextmanager
from loguru import logger
import traceback
import shutil
from src.utils.evaluation.evaluate_arguments import evaluate_sub_agent_history_f1
from src.utils.evaluation.evaluate_workflow_as_DAG import evaluate_workflow_multiple_runs
from src.utils.evaluation.eval_utils import (
    comprehensive_analysis,
    save_comprehensive_evaluation_results
)
from src.utils.model_factory import ModelFactory

def load_eval_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration from YAML file with environment variable overrides."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        config = apply_judge_env_overrides(config)
        return config
    except FileNotFoundError:
        logger.error(f"Evaluation config file not found: {config_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading evaluation config: {e}")
        return {}


def apply_judge_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to judge config."""
    judge_model = os.getenv("ORCHESTRATION_BENCH_JUDGE_MODEL")
    if not judge_model:
        return config

    if "judge" not in config:
        config["judge"] = {"model": {}, "generation_params": {}}
    if "model" not in config["judge"]:
        config["judge"]["model"] = {}
    if "generation_params" not in config["judge"]:
        config["judge"]["generation_params"] = {}

    config["judge"]["model"]["model"] = judge_model

    provider = os.getenv("ORCHESTRATION_BENCH_JUDGE_PROVIDER")
    if provider:
        config["judge"]["model"]["provider"] = provider

    base_url = os.getenv("ORCHESTRATION_BENCH_JUDGE_BASE_URL")
    if base_url:
        config["judge"]["model"]["base_url"] = base_url

    api_key = os.getenv("ORCHESTRATION_BENCH_JUDGE_API_KEY")
    if api_key:
        config["judge"]["model"]["api_key"] = api_key

    api_format = os.getenv("ORCHESTRATION_BENCH_JUDGE_API_FORMAT")
    if api_format:
        config["judge"]["model"]["api_format"] = api_format

    aws_access_key_id = os.getenv("ORCHESTRATION_BENCH_JUDGE_AWS_ACCESS_KEY_ID")
    if aws_access_key_id:
        config["judge"]["model"]["AWS_ACCESS_KEY_ID"] = aws_access_key_id

    aws_secret_access_key = os.getenv("ORCHESTRATION_BENCH_JUDGE_AWS_SECRET_ACCESS_KEY")
    if aws_secret_access_key:
        config["judge"]["model"]["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    aws_region = os.getenv("ORCHESTRATION_BENCH_JUDGE_AWS_REGION")
    if aws_region:
        config["judge"]["model"]["AWS_REGION"] = aws_region

    temperature = os.getenv("ORCHESTRATION_BENCH_JUDGE_TEMPERATURE")
    if temperature:
        config["judge"]["generation_params"]["temperature"] = float(temperature)

    max_tokens = os.getenv("ORCHESTRATION_BENCH_JUDGE_MAX_TOKENS")
    if max_tokens:
        config["judge"]["generation_params"]["max_tokens"] = int(max_tokens)

    top_p = os.getenv("ORCHESTRATION_BENCH_JUDGE_TOP_P")
    if top_p:
        config["judge"]["generation_params"]["top_p"] = float(top_p)

    logger.info(f"Applied judge env overrides: model={judge_model}")
    return config


# 💡 [타격 1 보완] AsyncContextManager를 통한 다양한 Provider 클라이언트 생명주기 완벽 보장
@asynccontextmanager
async def managed_judge_model(judge_config: Dict[str, Any]):
    model = None
    try:
        model = await ModelFactory.create_model(judge_config)
        yield model
    finally:
        if model:
            # 지원하는 종료 메서드 탐색 및 방어적 호출 (close, aclose 등)
            close_method = getattr(model, "close", None) or getattr(model, "aclose", None)
            if close_method:
                try:
                    result = close_method()
                    if asyncio.iscoroutine(result):
                        await result
                    logger.info("Successfully closed judge model resources.")
                except Exception as close_error:
                    logger.warning(f"Error closing judge model: {close_error}")


def load_evaluation_data(file_path: str) -> Dict[str, Any]:
    """Load evaluation data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"Evaluation data file not found: {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading evaluation data: {e}")
        return {}


# 💡 [추가 발견 보완] 원자적 파일 쓰기 (Atomic Write)로 손상된 JSON 방어
def atomic_write_json(path: str, data: Dict[str, Any]):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    fd, tmp_path = tempfile.mkstemp(dir=directory if directory else ".", prefix=".tmp_eval_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise e


async def evaluate_single_file(file_path: str, 
                               eval_config: Dict[str, Any], 
                               tool_descriptions: Dict[str, Any], 
                               save_llm_results: bool = False, 
                               llm_results_dir: str = None,
                               judge_model: Any = None) -> Dict[str, Any]:
    """Evaluate a single output file with both DAG and arguments evaluation."""
    logger.info(f"Evaluating file: {file_path}")
    agent_change_weight = eval_config.get("evaluation_params", {}).get("agent_change_weight", 0.8)
    status_change_weight = eval_config.get("evaluation_params", {}).get("status_change_weight", 0.5)
    
    data = load_evaluation_data(file_path)
    if not data:
        return {"error": f"Failed to load data from {file_path}", "file_path": file_path}
    
    try:
        # Workflow DAG evaluation
        logger.info("Running workflow DAG evaluation...")
        try:
            dag_results = evaluate_workflow_multiple_runs(data, agent_change_weight=agent_change_weight, status_change_weight=status_change_weight)
            if not isinstance(dag_results, dict):
                dag_results = {"error": f"Unexpected return type: {type(dag_results)}"}
        except Exception as e:
            logger.warning(f"Workflow DAG evaluation failed: {e}")
            dag_results = {"error": str(e)}
        
        # Arguments evaluation
        logger.info("Running arguments evaluation...")
        if eval_config:
            try:
                file_identifier = os.path.splitext(os.path.basename(file_path))[0] if save_llm_results else None

                args_results = await evaluate_sub_agent_history_f1(
                    data, 
                    eval_config,
                    tool_descriptions,
                    judge_model=judge_model,
                    save_llm_results=save_llm_results,
                    output_dir=llm_results_dir,
                    file_identifier=file_identifier
                )
                                
                if not isinstance(args_results, dict):
                    args_results = {"error": f"Unexpected return type: {type(args_results)}"}
                    
            except Exception as e:
                logger.warning(f"Arguments evaluation failed: {e}\n{traceback.format_exc()}")
                args_results = {"error": str(e)}
        else:
            args_results = {"error": "No evaluation config provided"}

        combined_results = {
            "file_path": file_path,
            "workflow_evaluation_with_failure": dag_results.get("total_score_with_failure", 0.0),
            "workflow_evaluation_without_failure": dag_results.get("total_score_without_failure", 0.0),
            "average_structural_GED": dag_results.get("average_structural_GED", 0.0),
            "average_component_GED": dag_results.get("average_component_GED", 0.0),
            "failed_workflow_generation": dag_results.get("failed_workflow_generation", 0),
            "total_comparison_steps_evaluated": dag_results.get("total_comparison_steps_evaluated", 0),
            "arguments_key_score": args_results.get("key_score", {}).get("avg_f1", 0.0) if isinstance(args_results, dict) else 0.0,
            "arguments_value_score_not_count_rejection": args_results.get("value_score_not_count_rejection", {}).get("avg_f1", 0.0) if isinstance(args_results, dict) else 0.0,
            "total_fc_successful_calls": args_results.get("total_fc_successful_calls", 0.0),
            "total_rejection_cases": args_results.get("total_rejection_cases", 0.0),
            "total_true_positive_reject": args_results.get("total_true_positive_reject", 0.0),
            "total_true_negative_reject": args_results.get("total_true_negative_reject", 0.0),
            "total_false_positive_reject": args_results.get("total_false_positive_reject", 0.0),
            "total_false_negative_reject": args_results.get("total_false_negative_reject", 0.0),
            "total_true_positive_fc": args_results.get("total_true_positive_fc", 0.0),
            "total_false_positive_fc": args_results.get("total_false_positive_fc", 0.0),
            "total_false_negative_fc": args_results.get("total_false_negative_fc", 0.0),  
            "total_true_negative_fc": args_results.get("total_true_negative_fc", 0.0),
            "total_function_call_cases": args_results.get("total_function_call_cases", 0.0),
            "function_name_accuracy": args_results.get("function_name_score", {}).get("f1-score", 0.0) if isinstance(args_results, dict) else 0.0,
            "detail": {
                "workflow_detail": dag_results,
                "arguments_detail": args_results
            }
        }

        if save_llm_results and llm_results_dir and file_identifier:
            save_comprehensive_evaluation_results(combined_results, llm_results_dir, file_identifier)

        logger.success(f"✅ Successfully evaluated: {file_path}")
        return combined_results

    except Exception as e:
        logger.error(f"Error evaluating {file_path}: {e}\n{traceback.format_exc()}")
        return {"error": str(e), "file_path": file_path}

async def evaluate_directory(directory_path: str, eval_config: Dict[str, Any], 
                           tool_descriptions: Dict[str, Any],
                           pattern: str = "*.json", sequential: bool = False,
                           output_path: str = None, save_llm_results: bool = False,
                           llm_results_dir: str = None,
                           start_over: bool = False,
                           judge_model: Any = None) -> Dict[str, Any]:
    """Evaluate all JSON files in a directory with resume capability and rate-limit safety."""
    logger.info(f"Evaluating directory: {directory_path}")
    
    temp_dir = os.path.join(os.path.dirname(output_path) if output_path else ".", ".temp_eval_results")
    if start_over:
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            if output_path and os.path.exists(output_path):
                os.remove(output_path) 
        except Exception: 
            pass

    search_pattern = os.path.join(directory_path, "**", pattern)
    json_files = glob.glob(search_pattern, recursive=True)
    
    if not json_files:
        logger.warning(f"No files found matching pattern: {search_pattern}")
        return {"error": "No files found", "pattern": search_pattern}
    
    logger.info(f"Found {len(json_files)} files to evaluate")
    
    existing_results = {}
    completed_files = set()
    if output_path and os.path.exists(output_path):
        existing_results = load_existing_results(output_path)
        completed_files = get_completed_files(existing_results)
        logger.info(f"Found {len(completed_files)} already completed files")
    
    incremental_results = load_incremental_results(temp_dir)
    for result in incremental_results:
        if "file_path" in result and "error" not in result:
            completed_files.add(result["file_path"])
    
    remaining_files = [f for f in json_files if f not in completed_files]
    logger.info(f"Need to evaluate {len(remaining_files)} remaining files")
    
    all_results = []
    if "detailed_results" in existing_results:
        all_results.extend(existing_results["detailed_results"])
    all_results.extend(incremental_results)
    
    if remaining_files:
        total_files = len(remaining_files)
        
        if sequential:
            logger.info(f"Starting sequential evaluation of {total_files} remaining files...")
            for i, file_path in enumerate(remaining_files, 1):
                logger.info(f"Processing file {i}/{total_files}: {os.path.basename(file_path)}")
                try:
                    result = await evaluate_single_file(
                        file_path, eval_config, tool_descriptions, 
                        save_llm_results, llm_results_dir, judge_model
                    )
                    all_results.append(result)
                    save_incremental_result(result, temp_dir, len(all_results))
                    logger.success(f"✅ Completed {i}/{total_files}: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Failed to evaluate {file_path}: {e}")
                    error_result = {"error": str(e), "file_path": file_path}
                    all_results.append(error_result)
                    save_incremental_result(error_result, temp_dir, len(all_results))
        else:
            # 💡 [데드락 방어] max_concurrent 상하한 안전장치 적용 (1 ~ 100)
            raw_concurrent = eval_config.get("max_concurrent", 5)
            max_concurrent = max(1, min(int(raw_concurrent), 100))
            semaphore = asyncio.Semaphore(max_concurrent)
            logger.info(f"Starting concurrent evaluation of {total_files} remaining files (Max Concurrent: {max_concurrent})...")
            
            # 💡 [타격 2 보완] 개별 작업 예외 격리 워커 정의
            async def worker(file_path: str, index: int):
                async with semaphore:
                    logger.info(f"Processing file {index}/{total_files}: {os.path.basename(file_path)}")
                    try:
                        return await evaluate_single_file(
                            file_path, eval_config, tool_descriptions, 
                            save_llm_results, llm_results_dir, judge_model
                        )
                    except Exception as worker_error:
                        logger.error(f"Worker failed for {file_path}: {worker_error}")
                        return {"error": str(worker_error), "file_path": file_path}

            tasks = [worker(file_path, i) for i, file_path in enumerate(remaining_files, 1)]
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    file_path = remaining_files[i]
                    if isinstance(result, Exception):
                        error_result = {"error": str(result), "file_path": file_path}
                        all_results.append(error_result)
                    else:
                        all_results.append(result)
                    save_incremental_result(all_results[-1], temp_dir, len(all_results))
                    
            except Exception as e:
                logger.error(f"Concurrent evaluation batch failed: {e}")
    
    successful_evals = [r for r in all_results if "error" not in r]
    failed_evals = [r for r in all_results if "error" in r]

    if successful_evals:
        avg_workflow_scores_with_failure = sum(r.get("workflow_evaluation_with_failure", 0) or 0 for r in successful_evals) / len(successful_evals)
        avg_workflow_scores_without_failure = sum(r.get("workflow_evaluation_without_failure", 0) or 0 for r in successful_evals) / len(successful_evals)
        avg_structural_GED = sum(r.get("average_structural_GED", 0) or 0 for r in successful_evals) / len(successful_evals)
        avg_component_GED = sum(r.get("average_component_GED", 0) or 0 for r in successful_evals) / len(successful_evals)
        avg_arguments_key_score = sum(r.get("arguments_key_score", 0) or 0 for r in successful_evals) / len(successful_evals)
        avg_arguments_value_score = sum(r.get("arguments_value_score_not_count_rejection", 0) or 0 for r in successful_evals) / len(successful_evals)
        total_rejection_cases = sum(r.get("total_rejection_cases", 0) or 0 for r in successful_evals) 
        total_true_positive_reject = sum(r.get("total_true_positive_reject", 0) or 0 for r in successful_evals) 
        total_true_negative_reject = sum(r.get("total_true_negative_reject", 0) or 0 for r in successful_evals) 
        total_false_positive_reject = sum(r.get("total_false_positive_reject", 0) or 0 for r in successful_evals) 
        total_false_negative_reject = sum(r.get("total_false_negative_reject", 0) or 0 for r in successful_evals) 
        total_true_positive_fc = sum(r.get("total_true_positive_fc", 0) or 0 for r in successful_evals)  
        total_false_positive_fc = sum(r.get("total_false_positive_fc", 0) or 0 for r in successful_evals)  
        total_false_negative_fc = sum(r.get("total_false_negative_fc", 0) or 0 for r in successful_evals)  
        total_true_negative_fc = sum(r.get("total_true_negative_fc", 0) or 0 for r in successful_evals)  
        total_rejection_type_mismatch = sum(r.get("total_rejection_type_mismatch", 0) or 0 for r in successful_evals)  
        total_failed_generation = sum(r.get("total_failed_generation", 0) or 0 for r in successful_evals)  
        total_function_call_cases = sum(r.get("total_function_call_cases", 0) or 0 for r in successful_evals) 
        avg_function_name_accuracy = sum(r.get("function_name_accuracy", 0) or 0 for r in successful_evals) / len(successful_evals)
        total_failed_workflow_generation = sum(r.get("failed_workflow_generation", 0) or 0 for r in successful_evals)
        total_fc_successful_calls = sum(r.get("total_fc_successful_calls", 0) or 0 for r in successful_evals)
        total_comparison_steps_evaluated = sum(r.get("total_comparison_steps_evaluated", 0) or 0 for r in successful_evals)
        
        overall_confusion_matrix = {
            "total_true_positive_reject": total_true_positive_reject,
            "total_true_negative_reject": total_true_negative_reject,
            "total_false_positive_reject": total_false_positive_reject,
            "total_false_negative_reject": total_false_negative_reject,
            "total_true_positive_fc": total_true_positive_fc,
            "total_false_positive_fc": total_false_positive_fc,
            "total_false_negative_fc": total_false_negative_fc,
            "total_true_negative_fc": total_true_negative_fc,
            "total_rejection_cases": total_rejection_cases,
            "total_function_call_cases": total_function_call_cases
        }
        
        overall_analysis = comprehensive_analysis(overall_confusion_matrix)

        reject_f1 = overall_analysis.get("reject_decision_performance", {}).get("f1", 0)
        fc_f1 = overall_analysis.get("fc_decision_performance", {}).get("f1", 0)
        call_rejection_acc = round((reject_f1 + fc_f1) / 2, 4)

        fc_acc = round((avg_function_name_accuracy + avg_arguments_key_score + avg_arguments_value_score) / 3, 4)
        plan_score = round(avg_workflow_scores_with_failure, 4)
        average_score = round((call_rejection_acc + fc_acc + plan_score) / 3, 4)
        
        overall_stats = {
            "total_files": len(json_files),
            "successful_evaluations": len(successful_evals),
            "failed_evaluations": len(failed_evals),
            "total_failed_workflow_generation": total_failed_workflow_generation,
            "total_comparison_steps_evaluated": total_comparison_steps_evaluated,
            "key_metrics": {
                "Average": average_score,
                "Call Rejection Classification Accuracy": call_rejection_acc,
                "FC": fc_acc,
                "Plan": plan_score
            },
            "score_details": {
                "workflow_scores": {
                    "workflow_evaluation_with_failure": round(avg_workflow_scores_with_failure, 4),
                    "workflow_evaluation_without_failure": round(avg_workflow_scores_without_failure, 4),
                    "average_structural_GED": round(avg_structural_GED, 4),
                    "average_component_GED": round(avg_component_GED, 4)
                },
                "function_call_scores": {
                    "function_name_accuracy": round(avg_function_name_accuracy, 4),
                    "arguments_key_score": round(avg_arguments_key_score, 4),
                    "arguments_value_score_not_count_rejection": round(avg_arguments_value_score, 4),
                    "overall_function_call_failed_rate": round((total_function_call_cases - total_fc_successful_calls)/total_function_call_cases, 4) if total_function_call_cases > 0 else 0.0,
                    "total_fc_successful_calls": total_fc_successful_calls,
                    "total_function_call_cases": total_function_call_cases
                },
                "call_rejection_scores": {
                    "total_rejection_cases": total_rejection_cases,
                    "reject_decision_performance": overall_analysis.get("reject_decision_performance", {}),
                    "fc_decision_performance": overall_analysis.get("fc_decision_performance", {}),
                    "error_patterns": overall_analysis.get("error_patterns", {}),
                    "rejection_type_accuracy": round((total_true_positive_reject - total_rejection_type_mismatch) / total_true_positive_reject, 4) if total_true_positive_reject > 0 else 0.0,
                    "total_rejection_type_mismatch": total_rejection_type_mismatch,
                    "total_failed_generation": total_failed_generation
                },
                "raw_confusion_matrix": overall_confusion_matrix
            }
        }
    else:
        overall_stats = {
            "total_files": len(json_files),
            "successful_evaluations": 0,
            "failed_evaluations": len(failed_evals),
            "total_failed_workflow_generation": 0,
            "total_comparison_steps_evaluated": 0,
            "key_metrics": {"Average": 0.0, "Call Rejection Classification Accuracy": 0.0, "FC": 0.0, "Plan": 0.0},
            "score_details": {}
        }
    
    final_results = {
        "directory": directory_path,
        "overall_statistics": overall_stats,
        "detailed_results": all_results
    }
    
    if output_path:
        atomic_write_json(output_path, final_results)
        cleanup_temp_dir(temp_dir)
        
    return final_results

def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file using atomic write."""
    try:
        atomic_write_json(output_path, results)
        logger.success(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def load_existing_results(output_path: str) -> Dict[str, Any]:
    """Load existing evaluation results if available."""
    try:
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load existing results: {e}")
    return {}

def get_completed_files(existing_results: Dict[str, Any]) -> Set[str]:
    """Extract list of already completed files from existing results."""
    completed_files = set()
    if "detailed_results" in existing_results:
        for result in existing_results["detailed_results"]:
            if "file_path" in result and "error" not in result:
                completed_files.add(result["file_path"])
    return completed_files

def save_incremental_result(result: Dict[str, Any], temp_dir: str, file_index: int):
    """Save individual file result atomically for recovery purposes."""
    try:
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"result_{file_index:04d}.json")
        atomic_write_json(temp_file, result)
    except Exception as e:
        logger.warning(f"Could not save incremental result: {e}")

def load_incremental_results(temp_dir: str) -> List[Dict[str, Any]]:
    """Load all incremental results from temp directory."""
    results = []
    try:
        if os.path.exists(temp_dir):
            temp_files = sorted(glob.glob(os.path.join(temp_dir, "result_*.json")))
            for temp_file in temp_files:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    results.append(json.load(f))
            logger.info(f"Loaded {len(results)} incremental results from temp directory")
    except Exception as e:
        logger.warning(f"Could not load incremental results: {e}")
    return results

def cleanup_temp_dir(temp_dir: str):
    """Clean up temporary result files."""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary result files")
    except Exception as e:
        logger.warning(f"Could not clean up temp directory: {e}")

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive workflow evaluation tool")
    parser.add_argument("--input", required=True, help="Input file or directory path to evaluate")
    parser.add_argument("--agent-cards-path", default="data/EN/multiagent_cards", help="Path to agent card JSON files")
    parser.add_argument("--eval-config", default="config/base_config/eval_config.yaml", help="Path to eval config file")
    parser.add_argument("--output", help="Output file path for results")
    parser.add_argument("--pattern", default="*_out.json", help="File pattern to match")
    parser.add_argument("--mode", choices=["file", "directory"], default="auto", help="Evaluation mode")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent LLM API calls")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--skip-llm-eval", action="store_true", help="Skip LLM-based evaluation")
    parser.add_argument("--sequential", action="store_true", help="Process files sequentially")
    parser.add_argument("--save-llm-results", action="store_true", help="Save LLM inputs/outputs")
    parser.add_argument("--llm-results-dir", default="llm_evaluation_logs", help="Directory for LLM logs")
    parser.add_argument("--start-over", action="store_true", help="Start over from scratch")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    
    eval_config = load_eval_config(args.eval_config)
    if eval_config:
        eval_config["max_concurrent"] = args.max_concurrent
        eval_config["batch_size"] = args.batch_size
        eval_config["skip_llm_eval"] = args.skip_llm_eval
    
    try:
        if args.skip_llm_eval:
            # LLM 평가 생략 시 judge_model 없이 실행
            await run_evaluation_pipeline(args, eval_config, judge_model=None)
        else:
            judge_config = eval_config.get("judge", {}).get("model", {})
            if not judge_config.get("model"):
                logger.error("Judge model not specified in eval_config while LLM evaluation is enabled.")
                return
            
            # 💡 [핵심] AsyncContextManager를 통한 완벽한 Lifecycle 관리 적용
            async with managed_judge_model(judge_config) as judge_model:
                await run_evaluation_pipeline(args, eval_config, judge_model=judge_model)
                
    except Exception as e:
        logger.error(f"Evaluation failed: {e}\n{traceback.format_exc()}")


async def run_evaluation_pipeline(args, eval_config, judge_model):
    if args.mode == "auto":
        if os.path.isfile(args.input):
            mode = "file"
        elif os.path.isdir(args.input):
            mode = "directory"
        else:
            logger.error(f"Input path does not exist: {args.input}")
            return
    else:
        mode = args.mode
    
    tool_descriptions = {}
    agent_card_pathes = glob.glob(os.path.join(args.agent_cards_path, "*.json"))
    for path in agent_card_pathes:
        with open(path, 'r', encoding='utf-8') as f:
            agent_card = json.load(f)
        for tool in agent_card.get('tools', []):
            tool_descriptions[tool['name']] = tool['parameters']

    if mode == "file":
        logger.info(f"Evaluating single file: {args.input}")
        results = await evaluate_single_file(
            args.input, eval_config, tool_descriptions, 
            args.save_llm_results, args.llm_results_dir, judge_model
        )
        if "error" not in results and not args.output:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        elif args.output:
            save_results(results, args.output)
            
    elif mode == "directory":
        logger.info(f"Evaluating directory: {args.input}")
        default_output = os.path.join(args.input, f"{os.path.basename(args.input)}-result.json")
        output_path = args.output if args.output else default_output
        
        results = await evaluate_directory(
            args.input, eval_config, tool_descriptions,
            args.pattern, args.sequential, output_path, 
            args.save_llm_results, args.llm_results_dir,
            args.start_over, judge_model
        )

        if "error" not in results:
            stats = results["overall_statistics"]
            logger.info("📊 Overall Evaluation Summary:")
            logger.info(f"  Total Files: {stats['total_files']}")
            logger.info(f"  Successful: {stats['successful_evaluations']}")
            logger.info(f"  Failed: {stats['failed_evaluations']}")
            logger.info(f"  Key Metrics: {stats['key_metrics']}")


if __name__ == "__main__":
    asyncio.run(main())

최종 개선사항
✅ Judge Model 단순 close 호출 → AsyncContextManager 기반 Provider 독립적 Lifecycle 관리 전환
✅ asyncio.gather 무제한 실행 → Semaphore + max_concurrent 상한 검증으로 API Rate Limit 방어
✅ 단순 JSON 저장 → Atomic Write 적용으로 중간 종료 시 결과 파일 손상 방지
✅ 작업 단위 예외 처리 → Worker 격리 구조로 단일 실패가 전체 평가 파이프라인에 전파되지 않도록 개선
✅ CLI main 책임 집중 → run_evaluation_pipeline 분리로 실행 제어와 평가 로직 결합도 제거
✅ Judge 모델 타입 불명확 → Any 기반 호환 유지와 실제 반환 객체 중심 관리로 정적 분석 충돌 완화
✅ 임시 결과 저장 → Incremental Atomic 저장으로 Resume 복구 안정성 강화
✅ 동시성 설정값 검증 → 음수/과대 입력 차단으로 Deadlock 및 리소스 폭주 방지
✅ 파일 저장 실패 위험 → tempfile + os.replace 원자적 교체 방식으로 데이터 무결성 확보


이번 리팩토링으로 단순히 돌아가는 평가기가 아니라, 장애·누수·폭주까지 스스로 방어하는 엔터프라이즈급 Evaluation Engine으로 진화했으며, 남은 과제는 기능 개선이 아닌 최종 운영 하드닝 영역이다.
