import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import pystan
import os, sys

from sklearn.model_selection import train_test_split 
from pystan.diagnostics import check_div


predictors = ['Week','weeks_sinceILI','IsVax','IsILI_year','propReports_ontime','IsHealthworker','IsHousehold','IsHoliday',\
        'IsHHSymptom_year','weeks_sinceHHSymptom','IsSymptom_year',\
        'weeks_sinceSymptom','IsHHILI_year','weeks_sinceHHILI']

year = int(sys.argv[1])
samplesize=0.8
iterations= 5000
chains =int(sys.argv[2])
model = "pILIfourcdf"

tstart = time()

#############################
##Initialise dataset
with open("prevalence_reports11_17.csv") as rawdata:
    dfpredictors = pd.read_csv(rawdata,\
                    parse_dates=['SurveyWeek','SubmitDate','JoinDate'],\
                    index_col=['EpiYear','MasterID'],dtype={'MasterID':int})
    
with open("missed_preval_reports11_17.csv") as misseddata:
    dfmissed = pd.read_csv(misseddata,\
                    parse_dates=['SurveyWeek','SubmitDate','JoinDate'],\
                    index_col=['EpiYear','MasterID'],dtype={'MasterID':int})



###create design matrix
dfX = pd.concat([dfpredictors,dfmissed]) #sort=True not implemented in pandas 19.1
del dfpredictors
del dfmissed

#####################
##Filter to year#####
dfX = dfX.loc[year]
####################################
###Filter to take only MASTERS######
####################################
dfX = dfX.loc[dfX.IsMaster==1]

#####################
firstweek = dfX.Week.min()
lastweek = dfX.Week.max()
num_weeks = lastweek-firstweek+1
##Ensure index includes week number, but keep week number as predictor
dfX.set_index('Week', append=True, drop=False, inplace=True)
dfX.sort_index(inplace=True)

###General manipulation of the raw data
dfX['Int'] = np.ones((dfX.shape[0],1)).astype(int)
dfX['WeeksEnrolled'] = dfX.groupby(['MasterID'])['Int'].cumsum().astype(int) -1
dfX.loc[dfX.WeeksEnrolled==0,'WeeksEnrolled'] = 1

dfX['propReports_sofar'] = dfX.Reports_sofar/(dfX.WeeksEnrolled)
dfX['propReports_ontime'] = dfX.Reports_ontime/(dfX.WeeksEnrolled)
dfX['seasonWeek'] = dfX.Week - firstweek +1
dfX['ILI_week'] = dfX.ILI_week - firstweek+1
dfX.loc[dfX.ILI_week <0,'ILI_week'] = 0
dfX.loc[dfX.Symptom_week >0,'Symptom_week'] = \
dfX.loc[dfX.Symptom_week >0,'Symptom_week']  - firstweek+1
dfX.loc[dfX.HHSymptom_week > 0,'HHSymptom_week'] = \
dfX.loc[dfX.HHSymptom_week > 0,'HHSymptom_week'] - firstweek +1

dfX.loc[dfX.HHILI_week > 0,'HHILI_week'] = \
dfX.loc[dfX.HHILI_week > 0,'HHILI_week'] - firstweek +1

dfX['weeks_sinceILI'] = dfX.IsILIontime_year*(dfX.seasonWeek - dfX.ILI_week)/num_weeks
dfX['weeks_sinceSymptom'] = dfX.IsSymptom_year*(dfX.seasonWeek - dfX.Symptom_week)/num_weeks
dfX['weeks_sinceHHSymptom'] = dfX.IsHHSymptom_year*(dfX.seasonWeek - dfX.HHSymptom_week)/num_weeks
dfX.IsHHILI_year = dfX.IsHHILI_year.fillna(0)
dfX['weeks_sinceHHILI'] = dfX.IsHHILI_year*(dfX.seasonWeek - dfX.HHILI_week)/num_weeks

dfX.weeks_sinceHHSymptom.fillna(0, inplace=True)
dfX.IsHHSymptom_year.fillna(0, inplace=True)

#Users who don't know if they are healthcare workers are put into No 
dfX['IsHealthworker'] = (dfX.WorkWithPatients < 2).astype(int)
dfX['IsHousehold'] = (dfX.NumHousehold >1).astype(int)
dfX.IsHoliday = dfX.IsHoliday.astype(int)

dfX['IsILI_year'] = dfX.IsILIlate_year + dfX.IsILIontime_year - \
    dfX.IsILIlate_year*dfX.IsILIontime_year


##Remove late reports from response and put them into not reporting
dfX.loc[dfX.IsReport_late==1, 'Isfevercough'] = pd.np.NaN
########

##Ensure response is about household ILI and not just masters
dfX['IsHHILI_inc'] = dfX.Isfevercough + dfX.IsHouseILI \
- dfX.Isfevercough*dfX.IsHouseILI


#### Prior history constructed, now only fit to every week after first week
dfX = dfX.loc[dfX.Week > firstweek]


##Take random subset of Masters

sampleindex, test_index =  train_test_split(
    dfX.reset_index(level='Week',drop=True).reset_index().MasterID.unique(),
     test_size=1-samplesize, 
     shuffle =True,
     random_state=1)
sampleindex = [ (i,j) for i in sampleindex for j in range(firstweek+1,lastweek+1)]
test_index = [ (i,j) for i in test_index for j in range(firstweek+1,lastweek+1)]


#separate training from test data
dfXsample = dfX.loc[dfX.index.isin(sampleindex)]
dfX_test = dfX.loc[dfX.index.isin(test_index)]

#Take responses
#1 = report not ILI, 2= report with ILI, 3= not report
s_Cat = dfX.IsHHILI_inc.fillna(2)
s_Cat = s_Cat + 1
s_Cat = s_Cat.astype(int)

## Take housheold data for binomial pILI model
s_numHH = dfXsample.NumHousehold.astype(int)
s_numVax = dfXsample.NumVax.astype(int)
s_mrep = (dfXsample.Isfevercough==1).astype("int8")
s_HHILIexc = (dfXsample.IsHouseILI==1).astype("int8")
s_m_v = dfXsample.IsVax.astype("int8")

predictors = sorted(predictors)

##Filter predictor columns
dfXsample = dfXsample[predictors]
dfX_test = dfX_test[predictors]

###Modify datatypes to reduce memory usage
for dtype in ['float','integer']:
    col_change = dfX.select_dtypes(include=[dtype]).columns 
    dfX[col_change] = dfX.select_dtypes(include=[dtype]).apply(pd.to_numeric,downcast=dtype)


y_train =s_Cat.loc[s_Cat.index.isin( dfXsample.index)]
y_test = s_Cat.loc[s_Cat.index.isin( test_index)]
##create categorical 
dfRwks = pd.get_dummies(dfXsample.Week, prefix='Week', columns=['Week'], drop_first=False)
dfR = pd.get_dummies(dfXsample.Week, prefix='Week', columns=['Week'], drop_first=True)
dfR['Int'] = 1
dfR= dfR.reindex(columns=sorted(dfR.columns))

dfXsample.drop('Week', axis=1, inplace=True)

print("\nThe year selected is: %i\n" % year)

##Remove dfX from memory to free up RAM
del dfX


############################
#####Begin STAN model#######
############################
regression_code = """
functions {
    /**
    * User defined probability function that is an improper categorical
    * distribution, where the probability vector is not a simplex.
    * 
    * @param x_v Number of individiduals in house who are sick
    */
    real minicat_lpmf(int y, int n_v, int n_u, int m_v,
        int mrep, real pILI, real p1, real p3, real p4  ) {
        
        real log_ILI;
        real indpILI;
        real pmasterILI;

        if (y<3) {
            log_ILI = binomial_lpmf(0|n_v, pILI) + binomial_lpmf(0|n_u, pILI);
            if (y == 1) {
                // House does not have ILI
                return log(p1) + log_ILI;
            }
            else {
                // House has ILI
                indpILI = exp(binomial_lpmf(0|n_v,pILI) + binomial_lpmf(0|n_u, pILI));
                pmasterILI = (m_v*pILI + (1-m_v)*pILI) / (1- indpILI) ;

                if (mrep ==1){
                    // Master has ILI
                    return log(p4) + log1m(indpILI) +log( pmasterILI ); //+ log( pmGivenR  );
                    }
                else {
                    return log(p3) + log1m(indpILI) + log1m(pmasterILI); //+ log1m( pmGivenR) ;
                }
            }
        }
        else {
            // House did not report
            indpILI = exp(binomial_lpmf(0|n_v,pILI) + binomial_lpmf(0|n_u, pILI));
            pmasterILI = (m_v*pILI + (1-m_v)*pILI) / (1- indpILI) ;

            return log1m(
                p1* indpILI +
                ( p3*(1-pmasterILI) +
                p4* pmasterILI ) *(1- indpILI )
                );
            
        }
    }

}
data {
    int<lower=0> N; // Length of data
    int<lower=0> num_predictors;
    int<lower=1> num_weeks;
    int y[N]; // observations s_Cat
    int<lower=0> HH[N]; // Household size
    int<lower=0> HHv[N]; // Num Vax in HH
    int<lower=0,upper=1> mrep[N]; // Master is sick or not
    int<lower=0,upper=1> HHILIexc[N]; // HH exclusive sick
    int<lower=0,upper=1> m_v[N]; //Master is vaccinated or not
    matrix[N,num_weeks] Rwks; //Weeks dummy
    matrix[N,num_weeks] R; //categorical variables
}
parameters {
    vector[num_weeks] Week;
    vector[num_predictors] theta1;
    vector[2] betap;
    vector<lower=0,upper=1>[num_weeks] pILI; //estimates for pILI
}
model {
    vector[N] p;
    vector[N] p3;
    vector[N] p4;
    vector[N] phealexp;
    vector[N] pwk;
    vector[N] pwkv;
    vector[N] pwkexp;
    vector[N] wk;
    Week ~ normal(0,0.8367); //priors
    theta1 ~ normal(0,0.8367); //priors
	betap ~ normal(0,0.8367); //priors

    pwk = Rwks*pILI; //
    wk = R*Week; //week parameter
    p = inv_logit(X*theta1 + wk);
    p3 = inv_logit(X*theta1 + wk + betap[1]); //HH is sick
    p4 = inv_logit(X*theta1 + wk + betap[2]); // Master is sick
    for (n in 1:N){

        y[n] ~ minicat( HHv[n], HH[n]-HHv[n], m_v[n],
            mrep[n],pwk[n], p[n], p3[n], p4[n]);

    }
}
"""
log_reg_dict = {
    'N': dfXsample.shape[0],
    'num_predictors': dfXsample.shape[1],
    'num_weeks': num_weeks-1,
    'y': y_train.values,
    'HH': s_numHH.values,
    'HHv': s_numVax.values,
    'mrep': s_mrep.values,
    'HHILIexc': s_HHILIexc.values,
    'm_v': s_m_v.values,
    'Rwks': dfRwks.values,
    'R': dfR.values,
    'X': dfXsample.values,
}

#################################
tstantime = time()
sm = pystan.StanModel(model_code=regression_code,model_name=model) #store STAN model
print(model+ "Model creation complete... (after %3.2f seconds)" % (time()-tstantime))
print("Stan time begins...")
logfit = sm.sampling(data=log_reg_dict, iter=iterations, chains=chains,warmup=iterations-2500)

print("Stan complete at %.2f min" % (((time() -tstantime)/60)))
print(check_div(logfit, verbose = 3)) #greater than 2 gives extra diagnostic messages
samples = logfit.to_dataframe()

print(logfit, file=open('results/multinomial/prevalence/'+model+'ILI'+format(year)+'_sample_'+format(samplesize)+'N_'+format(iterations)+'chains_'+format(chains)+'.txt','a'))


y_test.to_csv('results/multinomial/prevalence/'+model+'_'+str(year)+'_sample'+format(samplesize)+'_N'+format(iterations//1000)+'k_test_set.csv')

df_results_theta1 = pd.DataFrame(samples[[col for col in samples.columns if 'theta1' in col]].values, columns = dfXsample.columns)
df_results_theta1 = df_results_theta1.add_prefix('theta1_')


df_results_beta = pd.DataFrame(samples[[col for col in samples.columns if 'betap' in col]].values, columns = ['HHILI','MasterILI'])


df_results_pILI = pd.DataFrame(samples[[col for col in samples.columns if 'pILI' in col]].values, \
        columns=['pILI_' +str(n) for n in range(firstweek+1,lastweek+1)])
df_results_weeks = pd.DataFrame(samples[[col for col in samples.columns if 'Week' in col]].values, \
        columns = ['Int']+['Week_'+ str(n) for n in range(firstweek+2,lastweek+1)]) 
df_results = df_results_theta1.join(df_results_beta.add_prefix('betap_'))
df_results = df_results.join(df_results_pILI)


df_results = df_results.join(df_results_weeks)
try:
    df_results.to_csv('results/multinomial/prevalence/'+model+'_logitILI'+format(year)\
    +'_sample_'+format(samplesize)+'N_'+format(iterations)\
    +'chains_'+format(chains)+'.csv')

except FileNotFoundError:
    os.mkdir('results/multinomial/prevalence')
    df_results.to_csv('results/multinomial/prevalence/'+model+'_logitILI'+format(year)\
    +'_sample_'+format(samplesize)+'N_'+format(iterations)+\
    'chains_'+format(chains)+'.csv')
    
