# Robot LLM Security

这是一个关于机器人LLM安全性的研究项目，主要包含对AI2-THOR环境中的机器人安全行为评估。

## 项目概述

本项目实现了对机器人在AI2-THOR虚拟环境中的安全行为评估系统，包括：

- 抽象层安全评估
- 详细层安全评估  
- 长期规划安全评估
- 消融实验和反馈机制

## 主要功能

### 1. 安全评估模块
- **抽象评估**: 评估机器人在高层任务中的安全决策
- **详细评估**: 评估机器人在具体操作中的安全行为
- **长期评估**: 评估机器人在长期任务规划中的安全性

### 2. 消融实验
- 支持不同安全机制的消融实验
- 包含反馈机制的对比实验

### 3. 结果分析
- 生成详细的评估报告
- 支持多种图表可视化
- 提供JSON格式的结果输出

## 文件结构

```
robot/
├── robot_llm_security_sample.py   # 示例评估脚本（已脱敏）
├── dataset/                       # 数据集目录
├── models/                        # 模型文件目录
├── hf_cache/                      # HuggingFace缓存
├── results/                       # 评估结果目录
│   ├── abstract_eval_result_*.json    # 抽象评估结果
│   ├── detailed_eval_result_*.json    # 详细评估结果
│   ├── long_eval_result_*.json        # 长期评估结果
│   └── chart_*.json                   # 图表数据
└── output*.log                    # 运行日志
```

## 环境要求

- Python 3.8+
- PyTorch
- AI2-THOR
- OpenAI API
- 其他依赖见代码中的import语句

## 使用方法

1. 安装依赖环境
2. 配置API密钥：
   - 复制 `robot_llm_security_sample.py` 为 `robot_llm_security.py`
   - 在 `robot_llm_security.py` 中替换以下占位符：
     - `your_deepseek_api_key` → 您的DeepSeek API密钥
     - `your_openai_api_key` → 您的OpenAI API密钥
     - `your_http_proxy` → 您的HTTP代理地址（如果需要）
     - `your_https_proxy` → 您的HTTPS代理地址（如果需要）
3. 运行评估脚本：

```bash
python robot_llm_security.py
```

## 评估结果

项目会生成多种评估结果文件，所有结果都保存在 `results/` 目录中：

- `results/abstract_eval_result_*.json`: 抽象层安全评估结果
- `results/detailed_eval_result_*.json`: 详细层安全评估结果  
- `results/long_eval_result_*.json`: 长期规划评估结果
- `results/chart_*.json`: 用于可视化的图表数据

## 注意事项

- 确保有足够的计算资源运行AI2-THOR环境
- 需要有效的OpenAI API密钥和DeepSeek API密钥
- 某些大文件（如模型缓存）已通过.gitignore排除
- **重要**：请勿将包含真实API密钥的文件提交到版本控制系统
- 使用示例文件 `robot_llm_security_sample.py` 作为模板，替换其中的占位符

## 许可证

请根据您的需要添加适当的许可证信息。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。
