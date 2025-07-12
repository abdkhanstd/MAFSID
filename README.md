# MAFSID: Multi-Agent Few-Shot Intrusion Detection for VANETs through Rapid Collaborative Learning

This repository contains the implementation for the paper "MAFSID: Multi-Agent Few-Shot Intrusion Detection for VANETs through Rapid Collaborative Learning".

## 🏗️ Architecture Overview

MAFSID employs a **5-agent collaborative system** where each agent specializes in different aspects of intrusion detection:

- **NetworkTrafficAnalyzer**: Analyzes network flow patterns
- **AnomalyDetector**: Detects statistical anomalies  
- **BehaviorAnalyzer**: Analyzes behavioral patterns
- **ProtocolAnalyzer**: Analyzes protocol-specific features
- **ThreatClassifier**: Classifies threat types

The agents communicate through **attention-based message passing** across 3 communication rounds per episode, enabling rapid collaborative learning in few-shot scenarios.

## 📋 Requirements

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn
```

**System Requirements:**
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

## 📂 Repository Structure

```
MASFID Final Paper/
├── train_agents.py          # Main training script for multi-agent system
├── few_shot_models.py       # Few-shot learning model implementations
├── multiclass_datasets.py   # Dataset loading and preprocessing
├── datasets/                # Dataset directory (see setup below)
├── models/                  # Pre-trained model weights
├── results/                 # Experiment results
└── graphs/                  # Result visualization scripts
```

## 🗂️ Dataset Setup

### Required Datasets

The experiments require the following datasets (contact authors for access):

1. **In-Vehicle Network Dataset**
2. **Car Hacking Dataset**
3. **CICIDS2017 Dataset**
4. **Public VANET Datasets**

### How to Organize Dataset Files

Organize your datasets as follows (matching the requirements in `multiclass_datasets.py`):

```
datasets/
├── CICIDS2017/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   └── [other CICIDS2017 files]
├── carhacking/
│   ├── car_hacking_data.csv
│   └── [other car hacking files]
├── In-Vehicle Network Intrusion Detection/
│   ├── in_vehicle_data.csv
│   └── [other in-vehicle files]
└── [other metadata files]
```

- For **CICIDS2017**, place `KDDTrain+.txt`, `KDDTest+.txt`, and any other required files in `datasets/CICIDS2017/`.
- For **Car Hacking**, place the main CSV and other files in `datasets/carhacking/`.
- For **In-Vehicle Network Intrusion Detection**, place the main CSV and other files in `datasets/In-Vehicle Network Intrusion Detection/`.
- For **Public VANET Datasets**, place metadata and related files in `datasets/`.

**Note**: Datasets and metadata files are not included in this repository. Please refer to the original sources or contact the authors.

## 🚀 Quick Start

### 1. Download Pre-trained Models

Download pre-trained weights and additional resources from:
**[SharePoint Link](https://stduestceducn-my.sharepoint.com/:f:/g/personal/201714060114_std_uestc_edu_cn/Er5fgl4DwUdDq9nu27a4-asBVAwhkXaAlvheepibtXHZ-Q?e=zwHc8v)**

Extract to the `models/` directory.

### 2. Run Multi-Agent Training

```bash
# Train the 5-agent collaborative system
python train_agents.py
```

This will:
- Initialize 5 specialized agents with communication modules
- Train on few-shot episodes (3-way, 5-way, 8-way scenarios)
- Save best models and individual agent weights
- Generate evaluation results

### 3. Monitor Training

The system will output:
- Communication round statistics
- Individual agent performance
- Ensemble accuracy improvements
- Best model checkpoints

## 📊 Reproducing Paper Results

### Evaluation Configurations

The system automatically tests multiple few-shot configurations:
- **3-way 1-shot** and **3-way 3-shot**
- **5-way 1-shot** and **5-way 3-shot**  
- **8-way 1-shot** and **8-way 3-shot**

### Expected Outputs

Results are saved to `results/` directory:
- `{dataset}_agent_based_results.txt` - Detailed performance metrics
- Individual agent performance breakdown
- Communication analysis and weights
- Confusion matrices and classification reports

### Key Metrics

- **Ensemble Accuracy**: Combined performance of all 5 agents
- **Individual Agent Performance**: Specialized agent contributions
- **Communication Effectiveness**: Message passing impact
- **Few-shot Learning Curves**: Performance vs. support examples

## 🔧 Customization

### Modify Agent Architecture

Edit `train_agents.py` to customize:
- Agent specializations (lines 76-125)
- Communication rounds (line 316)
- Training episodes (line 750)

### Adjust Few-shot Settings

Change episode creation parameters:
- `n_way`: Number of classes per episode
- `k_shot`: Support examples per class
- `q_query`: Query examples per class

## 📈 Visualization

Generate paper figures:
```bash
cd graphs/
python generate_paper_figures.py
```

## 🤖 Agent Communication Details

Each agent maintains:
- **Internal State**: Learned representation updated via communication
- **Communication History**: Record of 50 recent message exchanges  
- **Specialized Feature Extractor**: Domain-specific neural networks
- **Attention Mechanisms**: For prototype computation and message aggregation

## 💾 Model Outputs

Training produces:
- `best_agents.pth`: Complete 5-agent system
- `agent_{i}_{AgentType}.pth`: Individual agent weights
- Communication weight matrices
- Training statistics and loss curves


## ⚠️ Notes

- GPU recommended for faster training (CUDA auto-detected)

## 🛠️ Troubleshooting

**CUDA Out of Memory**: Reduce batch size in episode creation
**Dataset Loading Issues**: Verify dataset paths and file formats
**Communication Errors**: Check agent initialization and device placement

For additional support, please refer to the paper or contact the authors.
