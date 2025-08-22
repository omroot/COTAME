# Commitment of Traders (COT) Analysis and Modeling Engine (COTAME)

This is a repo that aims at now/forecasting for COT positioning for the following asset classes:
- WTI
- Brent
- HO
- Rbob 

For each asset class, we perform positioning modeling of the following two types of investors:
- Commercials
- Managed Money 

Two variations of target/response variables are used:
- raw long/short/net positioning as reported in the COT weekly report
- scaled long/short/net positioning by open interest 

The features used are computed from the following sources:
- prior COT positioning (raw and scaled by open interest) changes
- prior open interest changes 
- prior front  futures and spread volume changes 
- prior  front  futures rolled price changes 
- prior  front  futures rolled price volatility 
- prior synthetic spread change

For each asset class/investor type, and modeling horizon (nowcasting/forecasting) we run a machine learning pipeline that includes:
- Preliminary Exploratory Data Analysis (EDA) that focuses primarily on the statistical properties of the response variables
- Feature selection for the six repsonse variables: raw and OI scaled long/short/net positioning 
- Model selection combined with hyperparameters funetuning to essentially select the best model along with its hyper-parameters 
- XAI layer that leverages SHAP values to list what features are driving the final selected and trained model 

For example, for WTI Commercials forecating, the modeling notebooks are:
- `notebooks\cot_modeling\wti\wti_cot_comm_forecast_01_eda.ipynb`
- `notebooks\cot_modeling\wti\wti_cot_comm_forecast_02_feature_selection.ipynb`
- `notebooks\cot_modeling\wti\wti_cot_comm_forecast_03_model_selection.ipynb`
- `notebooks\cot_modeling\wti\wti_cot_comm_forecast_04_final_model.ipynb`


Modeling notebooks with results can be found in:
- ho: `notebooks\cot_modeling\ho\`
- rbob: `notebooks\cot_modeling\rbob\`
- br: `notebooks\cot_modeling\br\`
- wti: `notebooks\cot_modeling\wti\`


Across all feature and model selection, we use as performance metric the correlation between predicted and actual positioning. 