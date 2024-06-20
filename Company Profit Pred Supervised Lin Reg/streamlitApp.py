import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
regression_model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict profit based on input factors
def predict_profit(RD_Spend, Administration, Marketing_Spend):
    # Prepare the input data with the same number of features as training data
    input_data = np.array([[RD_Spend, Administration, Marketing_Spend]])
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    predicted_profit = regression_model.predict(input_data_scaled)[0]
    return predicted_profit

# Streamlit app
st.title('Profit Prediction App Calculations Greater than Lakhs')

# Input fields
RD_Spend = st.number_input('Enter R&D Spend:')
Administration = st.number_input('Enter Administration Spend:')
Marketing_Spend = st.number_input('Enter Marketing Spend:')

# Predict button
if st.button('Predict Profit'):
    predicted_profit = predict_profit(RD_Spend, Administration, Marketing_Spend)
    if predicted_profit > 0:
        st.success(f'Predicted Profit: {predicted_profit:.2f}')
    else:
        st.error(f'Predicted Loss: {abs(predicted_profit):.2f}')



# import numpy as np
# import pandas as pd
# import joblib

# df = pd.read_csv('1000_Companies.csv')
# df.drop('State', axis=1, inplace = True)

# from re import X
# x = x = df.drop('Profit', axis=1)
# y = df.Profit

# ## Model 
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.50, random_state = 42)

# # Standardizing the Data
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Implementation of Linear Regression
# from sklearn.linear_model import LinearRegression
# # Cross Validation
# from sklearn.model_selection import cross_val_score

# regression = LinearRegression()
# regression.fit(X_train, y_train)

# # Cross Validation Score
# meanSquaredError = cross_val_score(regression, X_train, y_train, scoring = 'neg_mean_squared_error', cv = 5)
# np.mean(meanSquaredError)

# # Making Predictions
# reg_pred = regression.predict(X_test)

# # Calculating R Squared Value
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, reg_pred)

# # Save the model and scaler
# joblib.dump(regression, 'linear_regression_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
