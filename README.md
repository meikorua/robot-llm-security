# Robot LLM Security

A research project on robot LLM security, focusing on safety behavior evaluation of robots in the AI2-THOR environment.

## Project Overview

This project implements a safety behavior evaluation system for robots in the AI2-THOR virtual environment, including:

- Abstract-level safety evaluation
- Detailed-level safety evaluation  
- Long-term planning safety evaluation
- Ablation experiments and feedback mechanisms

## Main Features

### 1. Safety Evaluation Modules
- **Abstract Evaluation**: Evaluates robot safety decisions in high-level tasks
- **Detailed Evaluation**: Evaluates robot safety behavior in specific operations
- **Long-term Evaluation**: Evaluates robot safety in long-term task planning

### 2. Ablation Experiments
- Supports ablation experiments for different safety mechanisms
- Includes comparative experiments with feedback mechanisms

### 3. Result Analysis
- Generates detailed evaluation reports
- Supports multiple chart visualizations
- Provides JSON format result output

## File Structure

```
robot/
├── robot_llm_security_sample.py   # Sample evaluation script (sanitized)
├── dataset/                       # Dataset directory
├── models/                        # Model files directory
├── hf_cache/                      # HuggingFace cache
├── results/                       # Evaluation results directory
│   ├── abstract_eval_result_*.json    # Abstract evaluation results
│   ├── detailed_eval_result_*.json    # Detailed evaluation results
│   ├── long_eval_result_*.json        # Long-term evaluation results
│   └── chart_*.json                   # Chart data
└── output*.log                    # Runtime logs
```

## Requirements

- Python 3.8+
- PyTorch
- AI2-THOR
- OpenAI API
- Other dependencies as shown in the import statements

## Usage

1. Install dependencies
2. Configure API keys:
   - Copy `robot_llm_security_sample.py` to `robot_llm_security.py`
   - Replace the following placeholders in `robot_llm_security.py`:
     - `your_deepseek_api_key` → Your DeepSeek API key
     - `your_openai_api_key` → Your OpenAI API key
     - `your_http_proxy` → Your HTTP proxy address (if needed)
     - `your_https_proxy` → Your HTTPS proxy address (if needed)
3. Run the evaluation script:

```bash
python robot_llm_security.py
```

## Evaluation Results

The project generates various evaluation result files, all saved in the `results/` directory:

- `results/abstract_eval_result_*.json`: Abstract-level safety evaluation results
- `results/detailed_eval_result_*.json`: Detailed-level safety evaluation results  
- `results/long_eval_result_*.json`: Long-term planning evaluation results
- `results/chart_*.json`: Chart data for visualization

## Important Notes

- Ensure sufficient computational resources to run the AI2-THOR environment
- Valid OpenAI API key and DeepSeek API key are required
- Some large files (such as model cache) are excluded via .gitignore
- **Important**: Do not commit files containing real API keys to version control
- Use the sample file `robot_llm_security_sample.py` as a template and replace the placeholders

## License

Please add appropriate license information as needed.

## Contributing

Issues and Pull Requests are welcome to improve this project.