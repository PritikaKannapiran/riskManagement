import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib


nav = st.sidebar.selectbox("Select the one you wish to predict!", ["Home", "Credit Risk", "Market Risk", "Stress Testing"])

if nav == "Home":
    st.title("Risk Management")
    st.markdown("<b>Open innovation for managing the risk profile</b>", unsafe_allow_html=True)
    st.image('riskManage.jpeg')
    
    st.markdown("You’ve found yourself in the right place if you’re looking to find if it’s \nsafe for the bank to issue a loan or if it’s risky")
    st.markdown("Head over to the navigation bar on the left of the screen to check for credit \nand market risks")

    st.markdown("<b>More about this project:</b>", unsafe_allow_html=True)
    st.markdown("The primary objective of this **Machine learning** project is to enable banks to \nmake data-driven decisions regarding **loan approvals** and **risk mitigation**. \nBy leveraging historical data, market indicators, and various risk factors, \nthe project aims to accurately assess creditworthiness and potential \nmarket risks associated with lending.")
    
    st.markdown("<b><u>Creators</u></b>", unsafe_allow_html=True)
    st.markdown("<b><u>Team Trex:</b> Anjali Kedia, Srijena Guin, Pritika Kannapiran", unsafe_allow_html=True)

if nav == "Credit Risk":
    st.title("Let's begin to calculate your Credit Risk!")
    
    inputAge_label = "Enter person's age:"
#     bold_inputAge_label = f"**{inputAge_label}**"
    age = st.number_input(inputAge_label)
#    st.write(age)
    
    inputIncome_label = "Enter person's income:"
#     bold_inputIncome_label = f"**{inputIncome_label}**"
    income = st.number_input(inputIncome_label)
#    st.write(income)
    
    st.markdown("<b>Select the appropriate house ownership type: </b>", unsafe_allow_html=True)
    st.markdown("Mortgage --> 0")
    st.markdown("Other    --> 1")
    st.markdown("Own      --> 2")
    st.markdown("Rent     --> 3")
    
    inputHouse_label = "Choose ownership type"
#     bold_inputHouse_label = f"**{inputHouse_label}**"
    house = st.selectbox(inputHouse_label, [0, 1, 2, 3])
    
    inputLength_label = "Enter person's employment length:"
#     bold_inputLength_label = f"**{inputLength_label}**"
    emp = st.number_input(inputLength_label)
    
    st.markdown("<b>Select the appropriate loan intent type: </b>", unsafe_allow_html=True)
    st.markdown("Debt consolidation --> 0")
    st.markdown("Education          --> 1")
    st.markdown("Home improvement   --> 2")
    st.markdown("Medical            --> 3")
    st.markdown("Personal           --> 4")
    st.markdown("Venture            --> 5")
    
    inputIntent_label = "Choose person's loan intent:"
#     bold_inputIntent_label = f"**{inputIntent_label}**"
    intent = st.selectbox(inputIntent_label, [0,1,2,3,4,5])
    
    st.markdown("<b>Select the appropriate loan grade type: </b>", unsafe_allow_html=True)
    st.markdown("A --> 0")
    st.markdown("B --> 1")
    st.markdown("C --> 2")
    st.markdown("D --> 3")
    st.markdown("E --> 4")
    st.markdown("F --> 5")
    st.markdown("G --> 6")
    
    inputGrade_label = "Choose person's loan grade:"
#     bold_inputGrade_label = f"**{inputGrade_label}**"
    grade = st.selectbox(inputGrade_label, [0,1,2,3,4,5,6])
    
    inputAmt_label = "Enter person's loan amount:"
#     bold_inputAmt_label = f"**{inputAmt_label}**"
    amt = st.number_input(inputAmt_label)
    
    inputInterest_label = "Enter person's interest rate amount:"
#     bold_inputInterest_label = f"**{inputInterest_label}**"
    interest = st.number_input(inputInterest_label)
    
    st.markdown("<b>Select the appropriate loan status type: </b>", unsafe_allow_html=True)
    st.markdown("Repaid --> 0")
    st.markdown("Pending --> 1")
    
    inputStatus_label = "Choose person's loan status:"
#     bold_inputStatus_label = f"**{inputStatus_label}**"
    status = st.selectbox(inputStatus_label, [0,1], index=1)
    
    inputPerInc_label = "Enter person's loan percent income:"
#     bold_inputPerInc_label = f"**{inputPerInc_label}**"
    per_inc = st.number_input(inputPerInc_label)
    
    st.markdown("<b>Select the appropriate type: </b>", unsafe_allow_html=True)
    st.markdown("No --> 0")
    st.markdown("Yes --> 1")
    
    inputPerDef_label = "Choose person's default:"
#     bold_inputPerDef_label = f"**{inputPerDef_label}**"
    per_def = st.selectbox(inputPerDef_label, [0,1], index=1)
    
    inputHist_label = "Enter person's credit history length:"
#     bold_inputHist_label = f"**{inputHist_label}**"
    hist = st.number_input(inputHist_label)
    
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
    st.markdown(" ")
    st.markdown("Assume, weight for each asset to be equal to 0.5")
    
    st.image('Zscore.jpeg')
    inputZscore_label = "Choose appropriate Z-Score by referring the above:"
#     bold_inputZscore_label = f"**{inputZscore_label}**"
    Zscore = st.selectbox(inputZscore_label, [1.645,1.96,2.33,2.575], index=1)
    
    inputPval_label = "Enter the portfolio value:"
#     bold_inputPval_label = f"**{inputPval_label}**"
    Pval = st.number_input(inputPval_label)
    
    inputLim_label = "Enter VAR limit:"
#    bold_inputLim_label = f"**{inputPval_label}**"
    Lim = st.number_input(inputLim_label)
    
    if st.button("Predict the risk",key = 9)==1:
         ## Order of passing the data into the pipeline:
#         cols=['ZScore', 'portfolio value']  ## List of columns of the original dataframe
#
#         input_data=[[Zscore, Pval]]
        
         pipe=joblib.load("std_dev.pkl")  ## Loading the pipeline
         st.markdown("<u><b>Standard Deviation</b></u>", unsafe_allow_html=True)
         pipeRound = round(pipe, 3)
         st.write(pipeRound)
         
         var = Zscore * Pval * pipeRound
         varRound = round(var, 3)
         st.markdown("<u><b>VAR:</b></u>", unsafe_allow_html=True)
#         st.markdown("VAR:")
         st.write(varRound)
         
         if Lim > varRound:
            st.markdown("<h1 style='font-size: 32px;'>NOT UNDER RISK!</h1>", unsafe_allow_html=True)
         if Lim < varRound:
            st.markdown("<h1 style='font-size: 32px;'>IT IS UNDER RISK!!</h1>", unsafe_allow_html=True)
            
if nav=="Stress Testing":
        st.title("Stress Testing")
        st.markdown("Stress testing is a risk management process that involves evaluating the performance of a financial system, organization, or portfolio under different adverse scenarios.")
        st.markdown("Banks can conduct stress tests to determine how their portfolio would perform under different market scenarios. This can help to identify potential areas of weakness and allow banks to take proactive measures to mitigate risk.")

        st.markdown("To perform stress testing, we need to follow these steps:")
        st.markdown("Let's assume we have a bank portfolio with 2 assets: Asset A & Asset B, with their respective weights of 60% and 40%. We want to stress test this portfolio under the scenario of an economic recession, which we assume will cause a 20% drop in the value of both assets.")
        st.markdown("Determine the stressed value of each asset: In our case, since we assume a 20% drop in the value of both assets, the stressed value of Asset A would be 80% of its original value, and the stressed value of Asset B would also be 80% of its original value.")
    
        st.markdown(" ")

        st.markdown("<u><b>Calculate the stressed portfolio value:</b></u>", unsafe_allow_html=True)
        st.markdown("The stressed portfolio value is simply the sum of the stressed values of each asset, multiplied by its weight in the portfolio. In our case, the stressed portfolio value would be-")
        st.markdown("<b>Stressed Portfolio Value = (0.6 * 80% * \$10,000) + (0.4 * 80% * \$15,000) = \$8,400 + \$9,600 = \$18,000</b>", unsafe_allow_html=True)

        st.markdown("<u><b>Calculate the portfolio loss:</b></u>", unsafe_allow_html=True)
        st.markdown("The portfolio loss is the difference between the original portfolio value and the stressed portfolio value. In our case, the portfolio loss would be-")
        st.markdown("<b>Portfolio Loss = \$25,000 - \$18,000 = \$7,000</b>", unsafe_allow_html=True)

        st.markdown("<u><b>Calculate the percentage portfolio loss:</b></u>", unsafe_allow_html=True)
        st.markdown("The percentage portfolio loss is simply the portfolio loss divided by the original portfolio value, multiplied by 100. In our case, the percentage portfolio loss would be-")
        st.markdown("<b>Percentage Portfolio Loss = (\$7,000 / \$25,000) * 100 = 28%</b>", unsafe_allow_html=True)

        st.markdown(" ")
        st.markdown("By performing stress testing, we can see that under the scenario of an economic recession causing a 20% drop in the value of both assets, the portfolio would experience a loss of 28%. This information can help the bank to make more informed decisions about their portfolio management and risk mitigation strategies.")
#         input_data=pd.DataFrame(input_data,columns=cols)  ## Converting input into a dataframe with respective columns
#
#         res=pipe.predict(input_data)[0]  ## Predicting the class
#         out={1:"RISKY to provide loan!", 0:"POSSIBLE to provide loan!"}
#         st.write(f"The Final Verdict obtained from the given model is that : {out[res]}")
