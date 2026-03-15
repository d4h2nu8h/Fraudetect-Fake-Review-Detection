# FRAUDetect: Fake Review Detection with Genetic Algorithms and SMOTE-Tomek Synergy

> A hybrid deep learning system that detects fake online reviews by combining Genetic Algorithm optimisation with LSTM, CNN, and DNN architectures — addressing class imbalance through SMOTE-Tomek resampling and deployed as a live web application via Flask and React.

*B.Tech Final Year Project — Vellore Institute of Technology (VIT), Chennai*

---

## Overview

An estimated 30% of online reviews are fraudulent, directly influencing consumer purchasing decisions and distorting market perceptions. Traditional rule-based and heuristic detection methods struggle to keep pace with the evolving and increasingly sophisticated tactics used by malicious actors. The challenge is compounded by class imbalance in training data — genuine reviews vastly outnumber fake ones — which causes standard classifiers to be biased toward the majority class.

FRAUDetect addresses both problems simultaneously. The system integrates Genetic Algorithm (GA) optimisation with three neural architectures — LSTM, CNN, and DNN — to automatically tune model hyperparameters and network structures for maximum detection accuracy. SMOTE-Tomek resampling corrects class imbalance before training, and Word2Vec embeddings provide semantically rich input representations. The final system is deployed as a React frontend backed by a Flask API, enabling users to paste any review text and receive an immediate real or fake prediction.

---

## Dataset

**Source:** YELPNYC Dataset — Kaggle (`Yelp_NYC_Metadata.csv`)

| Property | Detail |
|---|---|
| Platform | Yelp NYC restaurant reviews |
| Task type | Binary classification |
| Target variable | `Label` (1 = Fake, 0 = Not Fake) |
| Class distribution (before balancing) | 62.5% Fake, 37.5% Not Fake |
| Class distribution (after SMOTE-Tomek) | 50% / 50% |

**Features used:** `Review_id`, `Product_id`, `User_id`, `Rating`, `Date`, `Review` (raw text)

---

## Methodology

### Data Preprocessing

Raw review text was cleaned through three preprocessing steps before being passed to the word embedding layer:

- **Stemming** — words reduced to their root form to normalise vocabulary and reduce feature dimensionality
- **Stop word removal** — high-frequency, low-signal words eliminated to focus the model on meaningful content
- **Punctuation removal** — special characters stripped to reduce noise in the token space

### Data Balancing — SMOTE-Tomek

The dataset exhibited a 62.5/37.5 class split. Before model training, SMOTE-Tomek resampling was applied:

- **SMOTE** generates synthetic minority-class samples by interpolating between existing examples, augmenting the underrepresented class
- **Tomek links** identifies and removes overlapping boundary instances between classes, sharpening the decision boundary

The result is a perfectly balanced 50/50 class split used for all model training.

### Word Embedding — Word2Vec

Each review was converted into a dense vector representation using Word2Vec. Unlike bag-of-words or TF-IDF approaches, Word2Vec captures semantic relationships between words — synonyms, contextual similarity, and word meaning — providing a richer input signal for all three neural architectures.

### Model Architecture — Genetic Algorithm Optimisation

Genetic Algorithms were used to automatically optimise the architecture and hyperparameters of each model. The GA iteratively evolves candidate configurations, selecting and recombining the strongest performers across generations to maximise detection accuracy. Three models were independently optimised:

**GA-Optimised LSTM**
- Long Short-Term Memory networks are well suited to sequential text data, capturing long-range dependencies between words across a review
- The GA tuned hidden layer sizes, dropout rates, learning rate, and recurrent structure

**GA-Optimised CNN**
- Convolutional Neural Networks extract local n-gram patterns from the embedded review text, identifying short-range linguistic features indicative of fake reviews
- The GA tuned filter sizes, number of filters, pooling strategy, and fully connected layer dimensions

**GA-Optimised DNN**
- Deep Neural Networks serve as a fully connected classification baseline combining all extracted features
- The GA tuned layer depth, hidden unit counts, activation functions, and regularisation

### API Integration — Flask + ngrok

The trained models are served via a Flask REST API exposed through ngrok. The `/fake_review` endpoint receives review text as a POST request, passes it through the preprocessing and embedding pipeline, runs inference against the trained model, and returns a prediction. ngrok tunnels the locally running Flask server to a public URL accessible by the React frontend.

### Web Application — React

A React frontend provides the user-facing interface. Users enter review text into a form, submit it via the Predict Result button, and the application renders a clear Fake or Not Fake verdict in real time.

---

## Results

All three models were evaluated on a held-out test set. The GA-Optimised LSTM substantially outperformed both CNN and DNN baselines, achieving perfect classification across all metrics.

**Model Comparison:**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| GA-Optimised LSTM | 100% | 1.00 | 1.00 | 1.00 |
| GA-Optimised CNN | 74% | 0.75 | 0.74 | 0.74 |
| GA-Optimised DNN | 74% | 0.75 | 0.74 | 0.74 |

**GA-Optimised LSTM — Confusion Matrix:**

| | Predicted Not Fake | Predicted Fake |
|---|---|---|
| Actual Not Fake | 991 | 0 |
| Actual Fake | 0 | 1008 |

**GA-Optimised CNN — Confusion Matrix:**

| | Predicted Not Fake | Predicted Fake |
|---|---|---|
| Actual Not Fake | 701 | 188 |
| Actual Fake | 325 | 784 |

The LSTM's ability to model sequential dependencies across the full review text — rather than only local patterns (CNN) or flat feature combinations (DNN) — explains the performance gap. The GA's iterative hyperparameter search found a configuration that fully exploited the LSTM's sequential modelling capacity on this dataset.

---

## Limitations & Future Work

**Current Limitations:**

- The 100% LSTM accuracy, while reflected in the classification report, warrants caution — this may indicate overfitting on the specific YELPNYC dataset and may not generalise to reviews from other platforms or domains
- The system processes English-language reviews only; multilingual fake review detection is not supported
- The Flask + ngrok deployment is not production-grade; ngrok URLs are ephemeral and the server must be restarted for each session
- The model does not capture reviewer behavioural features (e.g., posting frequency, account age, review timing patterns) which have been shown to improve detection in ensemble-based systems

**Future Directions:**

- Integrate transformer-based models (BERT, RoBERTa) as an alternative to Word2Vec + LSTM to leverage contextual pre-training on large review corpora
- Extend to multi-platform datasets (Amazon, Google Maps, TripAdvisor) to validate cross-domain generalisation
- Incorporate reviewer behavioural and graph-based features alongside text features for a richer feature set
- Replace the ngrok tunnel with a proper cloud deployment (AWS, GCP, or Heroku) for stable, persistent access
- Explore online learning approaches to allow the model to continuously adapt to new patterns of fake review generation without full retraining

---

## How to Run This Project

### Prerequisites

```bash
Python 3.8+
Node.js 16+
```

### 1. Clone the Repository

```bash
git clone https://github.com/d4h2nu8h/fraudetect-fake-review-detection.git
cd fraudetect-fake-review-detection
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key libraries:

```bash
pip install tensorflow keras flask flask-cors gensim imbalanced-learn scikit-learn pandas numpy
```

### 3. Start the Flask Backend

```bash
python main_file.py
```

The server will run on `http://0.0.0.0:5000`. Ensure ngrok is configured and running to expose the endpoint publicly for the frontend to access.

### 4. Start the React Frontend

```bash
cd frontend
npm install
npm start
```

The app will open at `http://localhost:3000`. Enter a review in the text field and click Predict Result to classify it as Fake or Not Fake.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+, JavaScript |
| Deep Learning | TensorFlow, Keras |
| Optimisation | Genetic Algorithm (custom implementation) |
| NLP & Embeddings | Word2Vec (Gensim) |
| Data Balancing | SMOTE-Tomek (imbalanced-learn) |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| API Framework | Flask, flask-cors |
| Tunnelling | ngrok |
| Frontend | ReactJS |
| Dataset | YELPNYC — Kaggle |

---

## Author

Dhanush Sambasivam

[![GitHub](https://img.shields.io/badge/GitHub-d4h2nu8h-181717?style=flat&logo=github)](https://github.com/d4h2nu8h)

---

## License

This project was submitted in partial fulfillment of the requirements for the degree of Bachelor of Technology in Computer Science and Engineering at Vellore Institute of Technology, Chennai (April 2024).
