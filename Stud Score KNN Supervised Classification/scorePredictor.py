import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
classifier = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title('Student Exam Score Prediction')

# Input fields for Study Hours and Previous Exam Score
study_hours = st.number_input('Enter Study Hours:')
exam_score = st.number_input('Enter Previous Exam Score:')


# Predict button
if st.button('Predict Pass/ Fail'):
    prediction = classifier.predict(scaler.transform(np.array([[study_hours, exam_score]])))[0]
    if prediction == 1:
        st.success('Result: Pass')
    else:
        st.error('Result: Fail')


# Output accuracy (optional)
# st.write(f'Accuracy: {accuracy:.2f}')


# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix

# # Load data
# df = pd.read_csv('student_exam_data.csv')

# # Define features and target
# x = df.iloc[:, :-1]
# y = df.iloc[:, -1]

# # Train-test split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # Standardize data
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# # Define and train the KNN model
# classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')
# classifier.fit(x_train, y_train)

# # Save the trained model
# joblib.dump(classifier, 'knn_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')