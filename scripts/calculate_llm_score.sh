#!/bin/bash

export OPENROUTER_API_KEY="your_openrouter_key"
exp_name=$1
python src/evaluation/llm_judge.py src/evaluation/outputs/${exp_name}