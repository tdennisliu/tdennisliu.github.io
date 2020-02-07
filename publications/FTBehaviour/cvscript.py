import pandas as pd
import numpy as np
import sys
from scipy.stats import multinomial


## Open datasets
year = int(sys.argv[1])
model = sys.argv[2] #'freep' or 'normal'
sample = 0.8
iterations = 5000
chains =4

with open('results/multinomial/prevalence/'+str(model)+'_logitILI'+str(year)\
                +'_sample_'+str(sample)+'N_'+str(iterations)+'chains_'+str(chains)+'.csv') as file:
    df_theta_post = pd.read_csv(file, index_col=0)
    
df_post_logit= df_theta_post


with open('results/multinomial/prevalence/'+model+'_'+str(year)+'_sample'+str(sample)+
            '_N'+str(iterations//1000)+'k_test_set.csv') as test_set:
    df_test = pd.read_csv(test_set)
    df_test.columns = ['MasterID','Week','Actual']
    df_test.set_index(['MasterID','Week'], inplace = True)
        
    
df_theta_post = df_theta_post.reindex(columns=sorted(df_theta_post.columns))
print(year, model)



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
#####################

####################################
###Filter to take only MASTERS######
####################################
dfX = dfX.loc[dfX.IsMaster==1]

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

#####################
dfX = dfX.loc[dfX.index.isin(df_test.index)]

s_numHH = dfX.NumHousehold.astype(int)
s_numVax = dfX.NumVax.astype(int)
s_mrep = (dfX.Isfevercough==1).astype("int8")
s_HHILIexc = (dfX.IsHouseILI==1).astype("int8")
s_m_v = dfX.IsVax.astype("int8")

predictors = ['Week','weeks_sinceILI','IsVax','IsILI_year','propReports_ontime','IsHealthworker','IsHousehold','IsHoliday',\
        'IsHHSymptom_year','weeks_sinceHHSymptom','IsSymptom_year',\
        'weeks_sinceSymptom','IsHHILI_year','weeks_sinceHHILI']

predictors = sorted(predictors)
##Filter chosen columns
dfX = dfX[predictors]

###Modify datatypes to reduce memory usage
for dtype in ['float','integer']:
    col_change = dfX.select_dtypes(include=dtype).columns
    dfX[col_change] = dfX.select_dtypes(include=dtype).apply(pd.to_numeric,downcast=dtype)

#sort columns for consistency
dfX = dfX.reindex(columns=sorted(dfX.columns))
dfRwks = pd.get_dummies(dfX.Week, prefix='Week', columns=['Week'], drop_first=False)
dfR = pd.get_dummies(dfX.Week, prefix='Week', columns=['Week'], drop_first=True)
dfR['Int'] = 1

dfR = dfR.reindex(columns=sorted(dfR.columns))

dfX.drop('Week',axis=1, inplace=True)


## Sample posterior and simulate the participation and the obsrvred housheold prevalence y_tilde

def func_multi(row):
    num_trials = 1
    trial = multinomial.rvs(n=num_trials, p = row.values)
    trial = trial/num_trials
    return trial#[0],trial[1],trial[2],trial[3],trial[4]

num_sims = 1 #per sample
n_samples = 1000 #dfX.shape[0]
df_y_tilde = pd.DataFrame(columns=np.arange(firstweek+1,lastweek+1))
df_y_part = pd.DataFrame(columns=np.arange(firstweek+1,lastweek+1))

print(model + " simulated")
for n in range(n_samples):
    #Sample from posterior
    if n%100==0:
        print('Sample %i' % n)
    ###dataframes are structured as (samples by variables)
    df_theta_post = df_theta_post.sample(1, replace=True)
    df_theta = df_theta_post[[col for col in df_theta_post.columns if 'theta1' in col]]
    df_beta = df_theta_post[[col for col in df_theta_post.columns if 'betap' in col]]
    df_week = df_theta_post[['Int'] + ['Week_' +str(i) for i in range(firstweek+2,lastweek+1)]]
    df_pILI = df_theta_post[[col for col in df_theta_post.columns if 'pILI_' in col]]
    
    # Calculate log odds
    ###rows will be (i,wk) and columns = n_samples
    ## each sample from the posterior is used once for one row of the dataset
    logodds_theta = (dfX.values * df_theta.values).sum(axis=1) + (dfR.values * df_week.values).sum(axis=1)
    if model =='three' or model =='threebin':
        logodds_beta = logodds_theta + df_beta.values.flatten()
    else:
        if model =='pILIfourcdf' or model == 'fourgamma' or model =='fourcondcdf':
            ##Need more compartments for the logodds
            logodds_beta = logodds_theta + df_beta.values[0][0]
            logodds_beta2 = logodds_theta + df_beta.values[0][1]
            pr_beta2 = 1/(1+ np.exp(-1*logodds_beta2))
        else:
            logodds_beta = (dfX.values * df_beta.values).sum(axis=1) + (dfR.values * df_week.values).sum(axis=1)

    
    pr_theta = 1/(1+ np.exp(-1*logodds_theta))
    pr_beta = 1/(1+ np.exp(-1*logodds_beta))

    pILI = (dfRwks.values * df_pILI.values).sum(axis=1)
    
    pheal = (1-pILI)**(s_numHH.values-s_numVax.values) * (1- pILI)**(s_numVax.values)
    pmasterILI = (s_m_v.values * pILI + (1 - s_m_v.values)*pILI) / (1-pheal)

    p1 = pr_theta*pheal
    p2 = pr_beta*  (1- pheal) * (1-pmasterILI) #REPORT HHexc ILI
    p3 = pr_beta2*(1-pheal) * pmasterILI # REPORT Master ILI
    p4 = 1- p1-p2-p3 #Not report


    df_prob = pd.DataFrame({'p1':p1, 'p2':p2, 'p3':p3, 'p4':p4}, index=dfX.index)

    for i in range(num_sims):
        

        sim = df_prob.apply(func_multi, axis=1)
        sim = sim.apply(pd.Series)

        ## groupby week to get weekly totals, index = week, columns = [0,1,2]
        y_tilde = sim.groupby(['Week']).agg(sum)
        ## create naive estimate, index = week
        s_y_tilde = (y_tilde[1]+y_tilde[2])/(y_tilde[0]+y_tilde[1]+y_tilde[2]) ##Simulated HH naive (ontime) estimate

        df_y_tilde = df_y_tilde.append(s_y_tilde, ignore_index=True)
        df_y_part = df_y_part.append((y_tilde[0]+y_tilde[1]+y_tilde[2])/ y_tilde.sum(axis=1), ignore_index=True)

df_y_part.to_hdf('results/multinomial/prevalence/'+model+str(year) \
             + 'sim_sample_N'\
              +str(iterations//1000)+'k.h5', key='part')
df_y_tilde.to_hdf('results/multinomial/prevalence/'+model+str(year) \
             + 'sim_sample_N'\
              +str(iterations//1000)+'k.h5', key='cat')
