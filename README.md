# EMG-Based Grasp Classification for Prosthetic Control

## Abstract

Surface electromyography (sEMG) is widely used to decode motor intention in upper limb prosthetic control. However, robust real-time grasp classification remains challenging due to signal variability, noise, and the limited availability of labeled data. Moreover, most existing approaches focus on fine-grained gesture recognition, which often exceeds the functional requirements of practical prosthetic hands.

This project proposes a task-oriented, two-stage supervised learning framework for efficient EMG-based grasp classification. First, gesture-level pre-training learns expressive representations from multichannel EMG, formulated as spatiotemporal grayscale images. These features are then transferred to a downstream task that classifies grasps into functional finger groups. The framework is built on a lightweight EEGNet derivative with a tailored extension.

**Key Results:** Our method outperformed baseline approaches by 19% and 31%, while maintaining substantially fewer parameters than state-of-the-art CNNs. The pre-trained model demonstrates strong generalization and real-time feasibility for practical prosthetic control.

---

## Project Overview

This repository contains the implementation of a two-stage transfer learning framework for EMG-based grasp classification. The approach combines:

1. **Pre-training Stage**: Gesture-level representation learning from 23 movement classes
2. **Transfer Stage**: Fine-grained grasp classification into functional finger groups (4 classes)

### Key Features

- ✅ Two-stage transfer learning framework
- ✅ Lightweight EEGNet architecture with custom extensions
- ✅ Spatiotemporal feature extraction from multichannel sEMG signals
- ✅ Robust cross-subject evaluation (leave-one-subject-out)
- ✅ Real-time classification capability for prosthetic control
- ✅ Data augmentation strategies for limited labeled data
- ✅ Comprehensive evaluation on Ninapro DB5 dataset

---

## Dataset

### Ninapro DB5 (MICCAI 2026)

- **Subjects**: 10 participants
- **Channels**: Multichannel surface EMG recordings
- **Pre-training Classes**: 23 movement types
- **Target Classes**: 4 functional grasp types (finger groups)
- **Evaluation**: Leave-one-subject-out (LOSO) cross-validation

### Data Organization

```
emg_dataset/              # Raw individual subject data (10 subjects: p1-p10)
├── p1/                   # Subject 1
│   ├── class1/           # Grasp class 1 samples
│   ├── class2/           # Grasp class 2 samples
│   ├── class3/           # Grasp class 3 samples
│   └── class4/           # Grasp class 4 samples
├── p2/ ... p10/          # Additional subjects

emg_dataset_augmented/    # Augmented dataset with synthetic samples
├── train/
├── val/
└── test/

pretrain_dataset/         # Pre-training data with 23 movement classes
├── p1/
│   ├── movement1/ ... movement23/
├── p2/ ... p10/

dataset/                  # Processed train/val/test splits
├── train/
├── val/
└── test/
```

---

## Project Structure

```
emg_class/
├── README.md                              # This file
├── direct_training_method.py              # Baseline: Direct training on grasp labels
├── self-supervised_classification.py      # Self-supervised pre-training approach
├── emg_classification_abdullah.ipynb      # Main analysis notebook
│
├── pretrained_eegnet_best.h5              # Best pre-trained model
├── pretrained_eegnet_complete.h5          # Complete pre-trained model
├── pretrained_eegnet_final.h5             # Final pre-trained model
├── transfer_model_4class.h5               # Fine-tuned transfer learning model
│
├── dataset/                               # Processed train/val/test splits
│   ├── train/
│   ├── val/
│   └── test/
│
├── emg_dataset/                           # Raw subject-specific data (p1-p10)
├── emg_dataset_augmented/                 # Augmented dataset
├── pretrain_dataset/                      # Pre-training movement data
└── loss_weights_grid_search/              # Grid search results for loss weights
```

---

## Model Architecture

### EEGNet-based Framework

The architecture consists of two key components:

#### 1. Feature Extractor (Pre-training Stage)
- **Input**: Multichannel sEMG signals (converted to spatiotemporal grayscale images)
- **Layers**:
  - Temporal convolution for frequency filtering
  - Depthwise convolution for spatial feature extraction
  - Separable convolutions for parameter efficiency
- **Output**: Gesture-level feature representations (23 classes during pre-training)

#### 2. Classifier (Transfer Stage)
- **Input**: Pre-trained feature representations
- **Adaptation**: Fine-tuning on grasp classification task (4 functional finger groups)
- **Optimization**: Transfer learning with reduced training requirements

### Design Principles
- Lightweight architecture for real-time inference
- Spatiotemporal feature learning from multichannel EMG
- Parameter efficiency compared to standard CNNs
- Robust generalization across subjects

---

## Installation & Setup

### Requirements
- Python 3.7+
- TensorFlow/Keras
- NumPy, SciPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn (visualization)

### Installation

```bash
# Clone repository
git clone https://github.com/anonym-ai-robotics/emg_class.git
cd emg_class

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow keras numpy scipy pandas scikit-learn matplotlib seaborn jupyter
```

---

## Usage

### 1. Pre-training Stage

Train the gesture-level feature extractor on 23 movement classes:

```bash
python self-supervised_classification.py
```

**Outputs:**
- `pretrained_eegnet_best.h5` - Best performing pre-trained model
- Training logs and performance metrics

### 2. Transfer Learning Stage

Fine-tune the pre-trained model for grasp classification (4 functional groups):

```python
# In your training script
from tensorflow.keras.models import load_model

# Load pre-trained model
base_model = load_model('pretrained_eegnet_best.h5')

# Fine-tune on target grasp classification task
# ... training code ...
model.save('transfer_model_4class.h5')
```

### 3. Evaluation

Run leave-one-subject-out cross-validation:

```bash
python direct_training_method.py  # Baseline comparison
```

### 4. Analysis & Visualization

Use the Jupyter notebook for comprehensive analysis:

```bash
jupyter notebook emg_classification_abdullah.ipynb
```

---

## Performance Results

### Comparison with Baselines

| Method | Accuracy | Parameters |
|--------|----------|-----------|
| Two-Stage Transfer (Proposed) | **X%** | ~Y |
| Direct Training Baseline | X-19% | Z |
| Gesture-to-Group Mapping | X-31% | Z' |
| State-of-the-art CNN | X-δ% | Z'' |

**Key Achievements:**
- ✅ 19% improvement over direct training
- ✅ 31% improvement over gesture mapping baseline
- ✅ Fewer parameters than SOTA CNN approaches
- ✅ Robust leave-one-subject-out generalization

### Cross-Subject Generalization

The leave-one-subject-out (LOSO) evaluation demonstrates:
- Strong subject-independent performance
- Practical feasibility for real-world prosthetic deployment
- Consistent performance across different users

---

## Ablation Studies

The project includes ablation studies confirming:

1. **Necessity of Pre-training Stage**
   - Pre-trained features → better generalization
   - Transfer learning crucial for small-scale target tasks

2. **Architectural Design Components**
   - Depthwise convolutions → parameter efficiency
   - Temporal convolution order → frequency filtering effectiveness
   - Separable convolutions → spatial-channel feature interaction

3. **Data Augmentation Impact**
   - Synthetic sample generation improves robustness
   - Reduces sensitivity to signal variability

---

## Real-Time Feasibility

The lightweight EEGNet-based architecture enables:
- **Low Latency**: Suitable for prosthetic control requirements
- **Memory Efficiency**: Deployable on embedded systems
- **Inference Speed**: Real-time classification capability
- **Battery Life**: Reduced power consumption for wearable systems

---

## Experimental Protocol

### Cross-Validation Strategy
- **Method**: Leave-One-Subject-Out (LOSO)
- **Training**: 9 subjects
- **Testing**: 1 held-out subject
- **Repeats**: 10 folds (one per subject)

### Signal Processing
- **Input Format**: Multichannel sEMG → spatiotemporal grayscale images
- **Window Size**: [Specify window duration in ms]
- **Overlap**: [Specify overlap percentage]
- **Normalization**: [Describe normalization strategy]

---

## Data Augmentation

To address limited labeled data:
- Synthetic sample generation
- Rotation and scaling transformations
- Time-frequency augmentation
- Noise injection for robustness

See `emg_dataset_augmented/` for augmented samples.

---

## Contributing

Contributions are welcome! Areas for improvement:
- Additional baseline comparisons
- Multi-subject adaptive learning
- Online learning strategies
- Hardware deployment optimization
- Mobile app integration

Please submit pull requests or issues for discussions.

---

## Citation

If you use this code in your research, please cite the related paper:

```bibtex
@inproceedings{YourAuthor2026,
  title={Two-Stage Transfer Learning for EMG-Based Grasp Classification in Prosthetic Control},
  author={Your Author(s)},
  booktitle={MICCAI 2026},
  year={2026}
}
```

---

## Related Work & References

### Key References
- **EEGNet**: Lawhern et al., EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces
- **Ninapro Dataset**: Atzori et al., Electromyography data for non-invasive naturally-controlled robotic hand prosthetics
- **Transfer Learning in Biosignals**: Domain adaptation and fine-tuning strategies for EMG

### Related Datasets
- Ninapro DB1-DB8: Various EMG datasets for gesture recognition
- CIFAR10/ImageNet: CNN baseline comparisons

---

## Contact & Support

For questions or issues:
- GitHub Issues: [Create an issue](https://github.com/anonym-ai-robotics/emg_class/issues)
- Email: [contact information]

---

## License

[Specify your license - MIT, Apache 2.0, etc.]

---

## Acknowledgments

- Ninapro dataset contributors
- MICCAI 2026 organizers
- Prosthetic research community

---

**Last Updated**: May 2026
**Project Status**: Active Development
