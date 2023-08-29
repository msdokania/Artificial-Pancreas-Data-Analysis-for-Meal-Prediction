
import pandas as pd
import numpy as np

# Loading the database
db_cgm = pd.read_csv('CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
db_insulin = pd.read_csv('InsulinData.csv',low_memory=False)

# Saving the Date Time stamp in order to use it for sorting
db_cgm['date_time_stamp']=pd.to_datetime(db_cgm['Date'] + ' ' + db_cgm['Time'])
db_insulin['date_time_stamp']=pd.to_datetime(db_insulin['Date'] + ' ' + db_insulin['Time'])

# Removing the rows that have null Glucose Readings
db_cgm = db_cgm[~db_cgm['Sensor Glucose (mg/dL)'].isnull()]

# Extracting the first Date and Time when Auto mode starts in the Insulin data
db_insulin = db_insulin.sort_values(by='date_time_stamp',ascending=True)
auto_mode_start = db_insulin.loc[db_insulin['Alarm']=='AUTO MODE ACTIVE PLGM OFF'].iloc[0]['date_time_stamp']

# Consider only those data which have more than 80% of values of total 288 values
maxRows = db_cgm.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>0.8*288).dropna().index.tolist()
db_cgm = db_cgm.loc[db_cgm['Date'].isin(maxRows)]

# Dividing the CGM data into Manual and Auto data by using the time when Auto mode started in Insulin data
auto_mode_data = db_cgm.sort_values(by='date_time_stamp',ascending=True).loc[db_cgm['date_time_stamp']>=auto_mode_start]
auto_mode_data = auto_mode_data.set_index('date_time_stamp')
manual_mode_data = db_cgm.sort_values(by='date_time_stamp',ascending=True).loc[db_cgm['date_time_stamp']<auto_mode_start]
manual_mode_data = manual_mode_data.set_index('date_time_stamp')

# Consider only those data which have more than 80% of values of total 288 values
#maxRows_manual = manual_mode_data.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>0.8*288).dropna().index.tolist()
#manual_mode_data = manual_mode_data.loc[manual_mode_data['Date'].isin(maxRows_manual)]

#maxRows_auto = auto_mode_data.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>0.8*288).dropna().index.tolist()
#auto_mode_data = auto_mode_data.loc[auto_mode_data['Date'].isin(maxRows_auto)]

# Final Calculations -
hyperglycemia_overnight_manual = (manual_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hyperglycemia_critical_overnight_manual=(manual_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_overnight_manual=(manual_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_sec_overnight_manual=(manual_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv1_overnight_manual=(manual_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv2_overnight_manual=(manual_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hyperglycemia_daytime_manual=(manual_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hyperglycemia_critical_daytime_manual=(manual_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_daytime_manual=(manual_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_sec_daytime_manual=(manual_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv1_daytime_manual=(manual_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv2_daytime_manual=(manual_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hyperglycemia_wholeday_manual=(manual_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hyperglycemia_critical_wholeday_manual=(manual_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_wholeday_manual=(manual_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_sec_wholeday_manual=(manual_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv1_wholeday_manual=(manual_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv2_wholeday_manual=(manual_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hyperglycemia_overnight_automode=(auto_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hyperglycemia_critical_overnight_automode=(auto_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_overnight_automode=(auto_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_sec_overnight_automode=(auto_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv1_overnight_automode=(auto_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv2_overnight_automode=(auto_mode_data.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hyperglycemia_daytime_automode=(auto_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hyperglycemia_critical_daytime_automode=(auto_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_daytime_automode=(auto_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_sec_daytime_automode=(auto_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv1_daytime_automode=(auto_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv2_daytime_automode=(auto_mode_data.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hyperglycemia_wholeday_automode=(auto_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hyperglycemia_critical_wholeday_automode=(auto_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_wholeday_automode=(auto_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

range_sec_wholeday_automode=(auto_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv1_wholeday_automode=(auto_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

hypoglycemia_lv2_wholeday_automode=(auto_mode_data.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

# Save the results
Results = pd.DataFrame({
'hyperglycemia_overnight':[ hyperglycemia_overnight_manual.mean(axis=0),hyperglycemia_overnight_automode.mean(axis=0)],
'hyperglycemia_critical_overnight':[ hyperglycemia_critical_overnight_manual.mean(axis=0),hyperglycemia_critical_overnight_automode.mean(axis=0)],
'range_overnight':[ range_overnight_manual.mean(axis=0),range_overnight_automode.mean(axis=0)],
'range_sec_overnight':[ range_sec_overnight_manual.mean(axis=0),range_sec_overnight_automode.mean(axis=0)],
'hypoglycemia_lv1_overnight':[ hypoglycemia_lv1_overnight_manual.mean(axis=0),hypoglycemia_lv1_overnight_automode.mean(axis=0)],
'hypoglycemia_lv2_overnight':[ np.nan_to_num(hypoglycemia_lv2_overnight_manual.mean(axis=0)),np.nan_to_num(hypoglycemia_lv2_overnight_automode.mean(axis=0))],
'hyperglycemia_daytime':[ hyperglycemia_daytime_manual.mean(axis=0),hyperglycemia_daytime_automode.mean(axis=0)],
'hyperglycemia_critical_daytime':[ hyperglycemia_critical_daytime_manual.mean(axis=0),hyperglycemia_critical_daytime_automode.mean(axis=0)],
'range_daytime':[ range_daytime_manual.mean(axis=0),range_daytime_automode.mean(axis=0)],
'range_sec_daytime':[ range_sec_daytime_manual.mean(axis=0),range_sec_daytime_automode.mean(axis=0)],
'hypoglycemia_lv1_daytime':[ hypoglycemia_lv1_daytime_manual.mean(axis=0),hypoglycemia_lv1_daytime_automode.mean(axis=0)],
'hypoglycemia_lv2_daytime':[ hypoglycemia_lv2_daytime_manual.mean(axis=0),hypoglycemia_lv2_daytime_automode.mean(axis=0)],
'hyperglycemia_wholeday':[ hyperglycemia_wholeday_manual.mean(axis=0),hyperglycemia_wholeday_automode.mean(axis=0)],
'hyperglycemia_critical_wholeday':[ hyperglycemia_critical_wholeday_manual.mean(axis=0),hyperglycemia_critical_wholeday_automode.mean(axis=0)],
'range_wholeday':[ range_wholeday_manual.mean(axis=0),range_wholeday_automode.mean(axis=0)],
'range_sec_wholeday':[ range_sec_wholeday_manual.mean(axis=0),range_sec_wholeday_automode.mean(axis=0)],
'hypoglycemia_lv1_wholeday':[ hypoglycemia_lv1_wholeday_manual.mean(axis=0),hypoglycemia_lv1_wholeday_automode.mean(axis=0)],
'hypoglycemia_lv2_wholeday':[ hypoglycemia_lv2_wholeday_manual.mean(axis=0),hypoglycemia_lv2_wholeday_automode.mean(axis=0)]                         
},
index=['manual_mode','auto_mode'])

Results.to_csv('Results.csv',header=False,index=False)