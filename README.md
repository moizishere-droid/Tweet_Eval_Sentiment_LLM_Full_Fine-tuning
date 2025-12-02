# 🐦 Tweet Sentiment Classifier

A **task-specific fine-tuned Transformer model** for classifying the sentiment of tweets into **Negative, Neutral, or Positive**. This project demonstrates **full LLM fine-tuning**, dataset preprocessing, and deployment using Hugging Face Transformers and Datasets.

---
🌐 Model Link

You can view or download the fine-tuned model here:https://huggingface.co/Abdulmoiz123/Tweet_Eval_Sentiment_LLM_Full_Fine-tuning
Hugging Face: Tweet Sentiment Classifier

Users can load the model directly into their code using Hugging Face Transformers pipeline.

---

## 🚀 Highlights

* **Full Fine-Tuning of DistilBERT**: Adapted a pre-trained language model to a specific task (tweet sentiment classification).
* **Task-Specific Dataset**: Fine-tuned on the `tweet_eval` sentiment dataset with train/validation/test splits.
* **Efficient Tokenization**: Batched tokenization with truncation and padding for optimized GPU usage.
* **Performance Metrics**: Evaluated using **accuracy**, achieving **74.05%** on the validation set.
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

* The classifier can correctly predict the sentiment of tweets with **~74% accuracy** after 3 epochs.
* The project highlights my ability to:

  * Work with **pre-trained language models**
  * Adapt them to **domain-specific tasks**
  * Build a **reusable ML pipeline**


