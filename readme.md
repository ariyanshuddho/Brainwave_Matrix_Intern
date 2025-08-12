This project is part of my Brainwave Matrix Internship.
The goal of this task is to build a Fake News Detection system using Logistic Regression and TF-IDF Vectorization.

📌 Project Overview
The fake news detection system takes in a news headline or article and predicts whether it is Real or Fake.
The model was trained on the Kaggle Fake News Dataset, which contains both real and fake news articles.

🛠 Technologies Used
Python 3

scikit-learn

pandas

numpy

pickle (for model persistence)

📂 Project Structure
csharp
Copy
Edit
Brainwave_Matrix_Solution/
│
├── task1/
│   ├── main.py                  # Model training
│   ├── test_model.py             # Testing saved model
│   ├── requirements.txt          # Python dependencies
│   ├── data/
│   │   ├── fake.csv
│   │   ├── true.csv
│   │   ├── logistic_regression_model.pkl
│   │   └── tfidf_vectorizer.pkl
│   └── README.md                 # This file
📊 Model Performance
Accuracy: 94.6%

Class	Precision	Recall	F1-Score
Fake	0.93	0.95	0.94
Real	0.96	0.94	0.95

🚀 How to Run the Project
1️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
2️⃣ Train the model
bash
Copy
Edit
python main.py
3️⃣ Test with custom input
bash
Copy
Edit
python test_model.py
Example:

text
Copy
Edit
Enter news text: The prime minister announced a new economic policy today.
Prediction: Real News
📦 Requirements
All dependencies are listed in requirements.txt.

✍ Author
Shuddho – Brainwave Matrix Intern
GitHub Profile