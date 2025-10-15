🛳️ Titanic Survival Prediction

🎯 Objective

The goal of this project is to predict whether a passenger on the Titanic would survive or not based on features such as age, gender, class, fare, and other factors.
This project is part of CodSoft Data Science Internship - Task 1.

📂 Dataset

Dataset Name: Titanic-Dataset.csv

Source: Kaggle Titanic Dataset

The dataset contains details about passengers such as:

=>PassengerId

=>Pclass (Ticket class)

=>Name

=>Sex

=>Age

=>SibSp (Number of siblings/spouses aboard)

=>Parch (Number of parents/children aboard)

=>Fare

=>Embarked (Port of Embarkation)

=>Survived (Target variable: 0 = No, 1 = Yes)

⚙️ Technologies Used

=>Python

=>Pandas

=>NumPy

=>Scikit-learn

=>Streamlit

🧠 Model Building

 =>Data was cleaned (handled missing values, encoded categorical columns).
 =>Features were scaled using StandardScaler.
 =>A Logistic Regression model was trained to predict survival.
 =>The trained model and preprocessing pipeline were saved as titanic_pipeline.sav.

💻 How to Run the Project

1️⃣ Install Requirements:
pip install -r requirements.txt

2️⃣ Train the Model (if not already trained)
python train_and_save.py
This will create titanic_pipeline.sav inside the models/ folder.

3️⃣ Run the Streamlit App
streamlit run app.py
Then open the link shown in the terminal (usually http://localhost:8501)

📊 Output

=>The app allows users to input passenger details.

=>It predicts whether the passenger would survive or not survive.

🧾 Example Input

Feature	       Example Value
Pclass	           3
Sex	             Female
Age	               22
SibSp	             1
Parch            	 0
Fare	             7.25
Embarked	         S

✅ Predicted Result: Survived

🙌 Author
Madhurima Kalashikam
CodSoft Data Science Internship – October 2025
