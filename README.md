# Notification-Prioritization
# 📬 Notification Prioritisation Using ML & NLP

This project automates email classification using Machine Learning (ML) and Natural Language Processing (NLP), helping users efficiently manage email notifications by categorizing them into **Spam**, **High Priority**, or **Low Priority**. The system enhances productivity and email security by ensuring important emails are prioritized and malicious content is filtered.

---

## 🧠 Introduction

Handling a large volume of emails manually is time-consuming and error-prone. This project builds an ML-based system to:
- Automatically classify emails into **Spam**, **High Priority**, and **Low Priority**
- Reduce user workload and increase response time
- Improve communication management in real-time

---

## 🎯 Objectives

- Automate email prioritization using ML and NLP
- Enhance productivity by focusing on critical emails
- Protect users from spam and phishing content

---

## 🚧 Research Gap

- **Limited Contextual Understanding** in current models
- **Scalability Challenges** for high-volume environments
- **Lack of Adaptability** to evolving email patterns

---

## 🔍 Methodology

- **Synthetic Dataset**: Curated with realistic spam, high-priority, and low-priority emails
- **Preprocessing**: Text cleaning, lowercasing, stopword removal, TF-IDF vectorization
- **Modeling**:
  - **Logistic Regression** for spam detection
  - **Random Forest** for priority classification
- **Real-Time Monitoring**: Using IMAP to classify incoming emails every 60 seconds

---

## 💡 Proposed System

1. **Email Retrieval**: Connect to email server (e.g., Gmail) via IMAP
2. **Feature Extraction**: TF-IDF vectorization of subject and body
3. **Two-Stage Classification**:
   - Spam Detection: Logistic Regression
   - Priority Classification: Random Forest
4. **Real-Time Automation**: Classify emails continuously with minimal user intervention

---

## 📈 Results

| Metric                | Spam Detection | Priority Classification |
|-----------------------|----------------|--------------------------|
| **Accuracy**          | 95.67%         | 89.34%                   |
| **Precision**         | 94.52%         | 90.22%                   |
| **Recall**            | 93.86%         | 88.64%                   |
| **F1-Score**          | 94.19%         | 89.43%                   |

---

## ✅ Conclusion

The project demonstrates how ML and NLP can be applied for real-time email classification with:
- High spam detection accuracy
- Efficient notification prioritization
- Continuous background operation for live monitoring

---

## 🔮 Future Scope

- **Security Integration** with enterprise email systems
- **Workflow Automation** for high-priority routing
- **Scalability** for deployment across platforms (e.g., chatbots, CRM)

---

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Libraries**: `scikit-learn`, `nltk`, `pandas`, `imaplib`
- **Techniques**: TF-IDF, Logistic Regression, Random Forest
- **Protocol**: IMAP for email access

