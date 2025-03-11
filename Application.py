import numpy as np
import pickle
import streamlit as st


#loading the saved model 
#model = pickle.load(open('./ldpmodel.sav','rb'))
model = pickle.load(open('./tuned_model.sav','rb'))


    
def main():   
    
    #title 
    st.title("Loan Default Prediction System")
    st.markdown('Kindly enter the borrower\'s data')
    
    #getting user inputs
    income = st.text_input("Income of the borrower")
    business_or_commercial = st.text_input("Business or commercial. Enter 0 for business/commercial and 1 for Non-business/commercial-personal")
    Neg_ammortization = st.text_input("Negative amortization. Enter 0 for negative amortization and 1 for not negative")
    #Credit_Score = st.text_input("The credit score of the borrower")
    lump_sum_payment = st.text_input("Lump sum payment. Enter 0 for lump sum payment and 1 for not lump sum payment")
    #Num_Credit_Lines = st.text_input("The number of credit lines the borrower has open")
    credit_type = st.text_input("Credit type. Enter 0 for  Credit Information Bureau, 1 for Credit Reference Information Format, 2 for  Equipment financing or Equipment loan and 3 for Express loan or Expedited loan")
    co_applicant_credit_type = st.text_input("Co-application credit type. Enter 0 for Credit Information Bureau and 1 for Express loan or Expedited loan")
    #Has_Dependents = st.text_input("Whether the borrower had dependents - Yes: 1 or No: 0")
    submission_of_application = st.text_input("Submission channel. Enter 0 for Not institutional, 1 for Financial institution, and 2 for Not applicable")
         
    output = ''
#Credit_Score, Num_Credit_Lines, Has_Dependents,
    #button for prediction
    if st.button('Classify'):
        data = np.array([income, business_or_commercial, Neg_ammortization, lump_sum_payment, credit_type, co_applicant_credit_type, submission_of_application], dtype=np.float64)
        
        data_reshape = data.reshape(1, -1)
        
        #make prediction with the data
        result = model.predict(data_reshape)
        #result = int(result)
        
        if result == 0:
            output = 'Non-Defaulted'  
        else:
            output = 'Defaulted'
    
    st.info(output)
        
        
        
if __name__ == '__main__':
    main()
        