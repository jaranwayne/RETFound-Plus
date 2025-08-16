

# **RETFound Plus: A Time- and Person-Sensitive Foundation Model for Disease Prediction and Risk Stratification**

**RETFound Plus** is a time-aware and person-specific foundation model designed for disease prediction and risk stratification. It features strong transferability and interpretability, making it suitable for a wide range of medical AI applications.

---

## 📝 Key Features

- **Large-scale pretraining**: RETFound Plus is pre-trained using self-supervised learning on millions of longitudinal retinal images.
- **Comprehensive validation**: Proven effective for modeling both disease incidence and progression.
- **High adaptability**: Easily transferable to custom tasks across various clinical prediction scenarios.

---

## 🎉 Latest Updates

- 🧠 **August 2025**: The codebase and pre-trained model weights are now fully open-sourced!

---

## 🔧 Environment Setup

1. **Create a Conda environment**:

```bash
conda create -n rfp python=3.11.0 -y
conda activate rfp
```

2. **Install dependencies**:

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
git clone https://github.com/jaranwayne/RETFound-Plus.git
cd RETFound-Plus
pip install -r requirements.txt
```

---

## 🌱 Fine-Tuning RETFound Plus

### 1️⃣ Accessing Pretrained Weights


| Model Name         | Download Link | Source |
|--------------------|----------------|--------|
| RETFound_Plus   | [Access Here](https://drive.google.com/file/d/1ZYaY3AZS6Hmb32t3a3C_sWUHQpFWaEE6/view?usp=sharing) | TBD    |

---

### 2️⃣ Dataset Structure

Organize your data in the following directory format:

```
├── data_path
    ├── images/
        ├── image_1.jpg
        ├── image_2.jpg
        ├── ...
    ├── train.csv
    ├── val.csv
```

**Example of `train.csv`:**

| images     | tte | event |
|------------|-----|-------|
| image1.jpg | 1.7 | 1     |
| image2.jpg | 4.5 | 0     |
| image3.jpg | 3.8 | 1     |

**Example of `val.csv`:**

| images     | tte | event |
|------------|-----|-------|
| image4.jpg | 1.2 | 1     |
| image5.jpg | 2.5 | 0     |

> **Terminology**:  
> - `tte` (Time to Event): Time duration from baseline to the occurrence of a specific event.  
> - `event`: Binary indicator of whether the event occurred (1 = event occurred, 0 = censored).

---

### 3️⃣ Start Fine-Tuning

Use the following command to begin fine-tuning. Model checkpoints will be saved automatically, and evaluation will follow training:

```bash
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --savemodel \
    --batch_size 16 \
    --world_size 1 \
    --epochs 50 \
    --blr 2e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --data_path ./data_path \
    --input_size 224 \
    --task train_htn_incidence \
    --finetune ./retfound_plus_student_encoder.pth
```

---

## 📃 Citation

If you find this project helpful in your research, please cite the following paper:

```
TBD
```

