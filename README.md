# 🐦 Tweet Sentiment Classifier

A **task-specific fine-tuned Transformer model** for classifying the sentiment of tweets into **Negative, Neutral, or Positive**. This project demonstrates full LLM fine-tuning, dataset preprocessing, and deployment using Hugging Face Transformers and Datasets.

---

## 🚀 Highlights

* **Full Fine-Tuning of DistilBERT**: Adapted a pre-trained language model to a specific task (tweet sentiment classification).
* **Task-Specific Dataset**: Fine-tuned on the `tweet_eval` sentiment dataset with train/validation/test splits.
* **Efficient Tokenization**: Batched tokenization with truncation and padding for optimized GPU usage.
* **Performance Metrics**: Achieved **74.05% accuracy** on validation data after 3 epochs.
* **Easy Deployment**: Ready-to-use inference pipeline with Hugging Face `pipeline`.
* **GPU-Accelerated Training**: Supports mixed-precision (FP16) training for faster execution.

---

## 📈 Model Performance

| Metric          | Value  |
| --------------- | ------ |
| Evaluation Loss | 0.697  |
| Accuracy        | 74.05% |
| Epochs          | 3      |
| Samples/sec     | 925    |
| Steps/sec       | 57.8   |

> Achieved strong baseline performance on short, real-world tweet text data.

---

## 🛠 Technologies & Libraries

* **Python 3.x**
* **[Transformers](https://huggingface.co/docs/transformers/index)** – Pre-trained models & fine-tuning
* **[Datasets](https://huggingface.co/docs/datasets/)** – Loading and preprocessing datasets
* **[Evaluate](https://huggingface.co/docs/evaluate/)** – Metrics computation (accuracy)
* **PyTorch** – Model training backend

---

## ⚙️ Features

* **Train your own model**: Use custom datasets for fine-tuning on specific domains.
* **Batch & tokenize data efficiently** for GPU acceleration.
* **Evaluate & track metrics** automatically after each epoch.
* **Inference pipeline** ready for real-time tweet sentiment analysis.

---

## 🌐 Hugging Face Model Link

You can view or download the fine-tuned model here:
[**Tweet Sentiment Classifier - Hugging Face**](https://huggingface.co/Abdulmoiz123/Tweet_Eval_Sentiment_LLM_Full_Fine-tuning)

**Load directly in code:**

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Abdulmoiz123/Tweet_Eval_Sentiment_LLM_Full_Fine-tuning",
    tokenizer="Abdulmoiz123/Tweet_Eval_Sentiment_LLM_Full_Fine-tuning"
)

result = classifier("I love this product!")
print(result)
```
---

## 🌟 Key Learning & Focus

By building this project, I focused on:

* Understanding **full LLM fine-tuning** workflows
* Dataset handling and preprocessing with Hugging Face **Datasets**
* Efficient **tokenization** for Transformer models
* Using Hugging Face **Trainer** for training and evaluation
* Deploying **real-time inference pipelines** for NLP tasks
* Tracking **metrics & performance** to evaluate model quality

---

## 📌 Results

* The classifier predicts tweet sentiment with **~74% accuracy** after 3 epochs.
* Demonstrates my ability to:

  * Work with **pre-trained language models**
  * Adapt them to **domain-specific tasks**
  * Build a **reusable NLP pipeline**
