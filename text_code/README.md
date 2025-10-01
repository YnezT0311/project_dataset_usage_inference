# Dataset Usage Cardinality Inference (DUCI) on Text Data

## 📌 Overview

This folder provides code for **Dataset Usage Cardinality Inference (DUCI)** on text data, focusing on detecting **the extent to which a model has used a specific dataset** during training. The application example in this folder is **copyright infringement detection**, where an author or publishing house may want to determine whether and how much of their content has been incorporated into a model.

We conduct our experiments using **pre-trained GPT-2** and the **BookMIA dataset**, which contains:

- **50 new books** (published in 2023, ensuring they were not in GPT-2’s pretraining data).
- **50 old books** (memorized by ChatGPT, but uncertain for GPT-2).

To ensure a rigorous evaluation and prevent **data contamination** or **distribution shifts between members and non-members**, we only use the **50 new books** for evaluation.

---

## 🔧 Running the Code

### **1️⃣ Train Target and Reference Models**

To train **target models** and **reference models**, run:

```bash
bash train_ref_tar.sh
```

By default, this script:

- Trains **4 target models** on different proportions of the protected dataset.
- Trains **2 reference models** on unrelated data (to simulate the data from unknown sources).
- Supports different **sampling methods** for target model training:
  - **Random sampling** (`--sampling_type=random`): Randomly selects training sentences from each book.
  - **Sequential sampling** (`--sampling_type=sequential`): Uses the **first** $p$ proportion of each book.

---

### **2️⃣ Run Dataset Usage Cardinality Inference**

Once the models are trained, run:

```bash
bash duci.sh
```

This script:

- Computes **Membership Inference Attack (MIA) scores**.
- Performs **naïve MIA aggregation** for baselines
- Implements **our DUCI method**

---

## 📖 **Dataset and Partitioning**

We use the **50 new books** from the **BookMIA dataset** for evaluation:

- **Protected Pool**: 30 books, used in training.
- **Population Pool**: 20 books, not included in training.
- **Sentence-Level Entries**: Each sentence is treated as a separate unit.

Each target model is trained on a **random half** of each protected book, ensuring **non-overlapping splits**.

---

## 📊 **Inference Methodology**

The core idea is to determine **how much of a dataset was used** in model training by:

1. **Training reference models** on controlled subsets.
2. **Applying Membership Inference Attacks (MIA)** to detect dataset usage.
3. **Estimating dataset usage proportions** using DUCI, which refines MIA-based detection.
