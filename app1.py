import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import joblib


#with open('/Users/pritika/Desktop/TrEx/model.pkl', 'rb') as file:
#    model = pickle.load(file)

nav = st.sidebar.selectbox("Select the one you wish to predict!", ["Home", "Credit Risk", "Market Risk"])

if nav == "Home":
    st.title("Risk Management")
    st.markdown("**Open innovation for managing the risk profile**")
    st.image('/Users/pritika/Desktop/TrEx/riskManage.jpeg')
    
    st.markdown("You’ve found yourself in the right place if you’re looking to find if it’s \nsafe for the bank to issue a loan or if it’s risky")
    st.markdown("Head over to the navigation bar on the left of the screen to check for credit \nand market risks")
    
#    st.image('/Users/pritika/Desktop/TrEx/home2.jpeg')

    st.markdown("**More about this project:**")
    st.markdown("The primary objective of this **Machine learning** project is to enable banks to \nmake data-driven decisions regarding **loan approvals** and **risk mitigation**. \nBy leveraging historical data, market indicators, and various risk factors, \nthe project aims to accurately assess creditworthiness and potential \nmarket risks associated with lending.")
    
    st.markdown("**Creators**")
    st.markdown("**Team Trex:**  Anjali Kedia, Srijena Guin, Pritika Kannapiran")

if nav == "Credit Risk":
    st.title("Let's begin to calculate your Credit Risk!")
    
    inputAge_label = "Enter person's age:"
    bold_inputAge_label = f"**{inputAge_label}**"
    age = st.number_input(bold_inputAge_label)
#    st.write(age)
    
    inputIncome_label = "Enter person's income:"
    bold_inputIncome_label = f"**{inputIncome_label}**"
    income = st.number_input(bold_inputIncome_label)
#    st.write(income)
    
    st.text ("Select the appropriate house ownership type: ")
    st.text("Mortgage --> 0")
    st.text("Other    --> 1")
    st.text("Own      --> 2")
    st.text("Rent     --> 3")
    
    inputHouse_label = "Choose ownership type"
    bold_inputHouse_label = f"**{inputHouse_label}**"
    house = st.selectbox(bold_inputHouse_label, [0, 1, 2, 3])
    
    inputLength_label = "Enter person's employment length:"
    bold_inputLength_label = f"**{inputLength_label}**"
    emp = st.number_input(bold_inputLength_label)
    
    st.text ("Select the appropriate loan intent type: ")
    st.text("Debt consolidation --> 0")
    st.text("Education          --> 1")
    st.text("Home improvement   --> 2")
    st.text("Medical            --> 3")
    st.text("Personal           --> 4")
    st.text("Venture            --> 5")
    
    inputIntent_label = "Choose person's loan intent:"
    bold_inputIntent_label = f"**{inputIntent_label}**"
    intent = st.selectbox(bold_inputIntent_label, [0,1,2,3,4,5])
    
    st.text ("Select the appropriate loan grade type: ")
    st.text("A --> 0")
    st.text("B --> 1")
    st.text("C --> 2")
    st.text("D --> 3")
    st.text("E --> 4")
    st.text("F --> 5")
    st.text("G --> 6")
    
    inputGrade_label = "Choose person's loan grade:"
    bold_inputGrade_label = f"**{inputGrade_label}**"
    grade = st.selectbox(bold_inputGrade_label, [0,1,2,3,4,5,6])
    
    inputAmt_label = "Enter person's loan amount:"
    bold_inputAmt_label = f"**{inputAmt_label}**"
    amt = st.number_input(bold_inputAmt_label)
    
    inputInterest_label = "Enter person's interest rate amount:"
    bold_inputInterest_label = f"**{inputInterest_label}**"
    interest = st.number_input(bold_inputInterest_label)
    
    st.text ("Select the appropriate loan status type: ")
    st.text("Repaid --> 0")
    st.text("Pending --> 1")
    
    inputStatus_label = "Choose person's loan status:"
    bold_inputStatus_label = f"**{inputStatus_label}**"
    status = st.selectbox(bold_inputStatus_label, [0,1], index=1)
    
    inputPerInc_label = "Enter person's loan percent income:"
    bold_inputPerInc_label = f"**{inputPerInc_label}**"
    per_inc = st.number_input(bold_inputPerInc_label)
    
    st.text ("Select the appropriate type: ")
    st.text("No --> 0")
    st.text("Yes --> 1")
    
    inputPerDef_label = "Choose person's default:"
    bold_inputPerDef_label = f"**{inputPerDef_label}**"
    per_def = st.selectbox(bold_inputPerDef_label, [0,1], index=1)
    
    inputHist_label = "Enter person's credit history length:"
    bold_inputHist_label = f"**{inputHist_label}**"
    hist = st.number_input(bold_inputHist_label)
    
    if st.button("Predict Credit Score",key = 9)==1:
         ## Order of passing the data into the pipeline:
         cols=['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_status', 'loan_percent_income',
       'cb_person_default_on_file', 'cb_person_cred_hist_length']  ## List of columns of the original dataframe
                
         input_data=[[age, income, house, emp, intent, grade, amt, interest, status, per_inc, per_def, hist]]
        
         pipe=joblib.load("model.pkl")  ## Loading the pipeline
        
         input_data=pd.DataFrame(input_data,columns=cols)  ## Converting input into a dataframe with respective columns

         res=pipe.predict(input_data)[0]  ## Predicting the class
         out={1:"The Customer is capable of DEFAULTING. Hence it is RISKY to provide loan!", 0:"The Customer is capable of NOT DEFAULTING. Hence it is POSSIBLE to provide loan!"}
         st.write(f"The Final Verdict obtained from the given model is that : {out[res]}")
        
#    with open('/Users/pritika/Desktop/TrEx/model.pkl', 'rb') as file:
#        model = pickle.load(file)
#
#
#    if st.button('Predict'):
#        st.text("Prediction displays here")
#        prediction = model.predict([[age, income, house, emp, intent, grade, amt, interest, status, per_inc, per_def, hist]])
#        st.write("Prediction:", prediction)
    
if nav =="Market Risk":
    st.title("Let's begin to calculate your Market Risk!")
    
    st.image('/Users/pritika/Desktop/TrEx/Zscore.jpeg')
    inputZscore_label = "Choose appropriate Z-Score by referring the above:"
    bold_inputZscore_label = f"**{inputZscore_label}**"
    Zscore = st.selectbox(bold_inputZscore_label, [1.645,1.96,2.33,2.575], index=1)
    
    inputPval_label = "Enter the portfolio value:"
    bold_inputPval_label = f"**{inputPval_label}**"
    Pval = st.number_input(bold_inputPval_label)
    
    if st.button("Predict the risk",key = 9)==1:
         ## Order of passing the data into the pipeline:
         cols=['ZScore', 'portfolio value']  ## List of columns of the original dataframe
                
         input_data=[[Zscore, Pval]]
        
#         pipe=joblib.load("Newmodel.pkl")  ## Loading the pipeline
#
#         input_data=pd.DataFrame(input_data,columns=cols)  ## Converting input into a dataframe with respective columns
#
#         res=pipe.predict(input_data)[0]  ## Predicting the class
#         out={1:"RISKY to provide loan!", 0:"POSSIBLE to provide loan!"}
#         st.write(f"The Final Verdict obtained from the given model is that : {out[res]}")
