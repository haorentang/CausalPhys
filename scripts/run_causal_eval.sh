
nohup python3 evaluation/run_causal_tf_eval.py --model ALL --subfolder ALL --auto_eval --judge_model opengvlab/internvl3-78b --api_provider openrouter --api_key_env OPENROUTER_API_KEY --workers 10 >> logs/causal_eval_internvl3_$(date +%Y%m%d_%H%M%S).log 2>&1 &


