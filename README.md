# 📄 Document Triage API

**AI-powered API for semantic classification and triage of administrative documents using vector embeddings.**


## 🧠 Business Problem

Organizations responsible for handling **administrative documents**, such as appeals, requests, or formal defenses, often face a **high volume of unstructured text submissions** that must be reviewed by specialized teams.

These documents vary significantly in:

* Writing style
* Length
* Argument quality
* Level of formality

As demand increases, **manual triage becomes a bottleneck**, leading to:

* Longer response times
* Increased operational costs
* Inconsistent initial assessments
* Difficulty prioritizing cases effectively

Without an automated way to categorize incoming documents, all submissions are treated equally, regardless of their complexity or likelihood of acceptance, resulting in inefficient allocation of human resources.


## 🎯 Solution Overview

The Document Triage API provides an **AI-assisted classification layer** that analyzes the semantic content of administrative documents and assigns them to **predefined triage categories**, such as:

* `approved` (deferimento)
* `rejected` (indeferimento)
* `partially_approved` (parcial)

The system is designed to **support human decision-making**, not replace it.
Its primary purpose is to enable **faster routing, prioritization, and workflow automation**.


## 🏗️ How It Works

1. A client system sends the document text to the API
2. The API generates a semantic embedding using a pre-trained language model
3. The embedding is compared against reference examples using vector similarity search
4. The most relevant category is returned along with a confidence score

Example response:

```json
{
  "label": "rejected",
  "confidence": 0.91,
  "model_version": "v1"
}
```

## 💼 Business Value

By introducing an automated triage layer, organizations can:

* Reduce manual workload during initial document review
* Improve response times for high-volume processes
* Standardize initial categorization criteria
* Prioritize cases that require deeper human analysis
* Scale document processing without proportional increases in staff

This approach is particularly effective in **early-stage automation**, proof-of-concept environments, and hybrid AI–human workflows.


## ⚠️ Important Considerations

* The classification result is **not a final decision**
* All outputs are intended as **decision support signals**
* Final judgments remain the responsibility of human reviewers
* The system can be continuously improved using real historical data, subject to privacy and compliance requirements


## 🛠️ Technical Highlights

* RESTful API built with **FastAPI**
* Semantic embeddings via **Sentence Transformers**
* Vector similarity search (NumPy / FAISS)
* Stateless, containerized architecture
* Designed for low-latency inference and easy integration


## 🚀 Intended Use Cases

* Administrative appeal triage
* Legal or compliance document routing
* Support ticket classification
* Workflow prioritization systems
* AI-assisted document review pipelines


## 📌 Disclaimer

This project uses **synthetic data inspired by real-world administrative language**.
No confidential or personally identifiable information is included.
