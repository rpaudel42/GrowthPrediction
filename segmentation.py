# ******************************************************************************
# segmentation.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 3/26/19       Initial version,
# ******************************************************************************

import pandas as pd
import numpy as np

import dateutil
from datetime import datetime

#import visualization tools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mat
from mpl_toolkits.mplot3d import Axes3D

class Segmentation():
    def __init__(self):
        self.raw_data = pd.read_csv('customer.csv')
        self.transaction = self.raw_data

    def check_data(self):
        #check for null or mission value
        print("\n\nNull/Misisng Data :\n" , self.transaction.isnull().sum())

        print("\n\nTotal Transaction: ", len(self.transaction))
        print("Total Users:       ", len(set(self.transaction.user)))

    def get_rfm_metric(self):

        # get transaction date
        self.transaction['time'] = pd.to_datetime(self.transaction['time'])
        today = datetime.now()

        # Get Days since Recent Transaction (recency), Frequency of transaction and Amount spent per Seller
        self.transaction = self.transaction.groupby('user').agg(
            {'time': lambda date: (today - date.max()).days, 'user': ['count'],
             'amount_in_cents': ['sum']}).reset_index()
        self.transaction.columns = ['user', 'recency', 'frequency', 'monetary']

        #print(len(transaction))
        #print(transaction.describe())

    # get rank for each metrics
    def rank_r(self, x, p, t):
        if x <= t[p][0.25]:
            return str(1)
        elif x <= t[p][0.75]:
            return str(2)
        else:
            return str(3)

    def rank_f(self, x, p, t):
        if x <= t[p][0.75]:
            return str(3)
        else:
            return str(1)

    def rank_m(self, x, p, t):
        if x <= t[p][0.25]:
            return str(3)
        elif x <= t[p][0.75]:
            return str(2)
        else:
            return str(1)

    def define_rfm_segment(self, rows):
        if rows['rfm_score'] == '111':
            return 'best_users'
        elif rows['rfm_score'] == '211':
            return 'almost_lost'
        elif rows['rfm_score'] == '311':
            return 'lost_users'
        elif rows['rank_r'] == '3':
            return 'cheap_lost'
        elif rows['rank_f'] == '1':
            return 'loyal_users'
        elif rows['rank_m'] == '1':
            return 'big_spender'
        elif rows['rank_f'] == '3':
            return 'new_customer'
        else:
            return rows['rfm_score']

    def get_rfm_index(self):
        # Use 25 and 75th quantile to get rank for R, F and M ----
        threshold = self.transaction.drop('user', axis=1).quantile(q=[0.25, 0.75])
        threshold = threshold.to_dict()


        self.transaction['rank_r'] = self.transaction['recency'].apply(self.rank_r, args=('recency', threshold))
        self.transaction['rank_f'] = self.transaction['frequency'].apply(self.rank_f, args=('frequency', threshold))
        self.transaction['rank_m'] = self.transaction['monetary'].apply(self.rank_m, args=('monetary', threshold))
        self.transaction['rfm_score'] = self.transaction['rank_r'] + self.transaction['rank_f'] + self.transaction['rank_m']

        # Define segment based on RFM score
        self.transaction['segment'] = self.transaction.apply(self.define_rfm_segment, axis=1)

    def save_rfm_segment_pie_chart(self):
        transaction_count = self.transaction.groupby('segment', as_index=False).count()[['segment', 'user']]
        plt.pie(data=transaction_count, x='user', labels='segment', autopct='%1.1f%%',
                colors=mat.cm.Paired(np.arange(7) / 7.))
        plt.title('User segments and their distribution')
        #plt.show()
        plt.savefig('figures/segment_pie_char.png')

    def save_rfm_segment_scatter_plot(self):
        segment = {'best_users': 0, 'lost_users': 1, 'new_customer': 2, 'loyal_users': 3, 'cheap_lost': 4,
                   'big_spender': 5, 'almost_lost': 6}
        label = [segment[item] for item in self.transaction['segment']]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax = Axes3D(fig)
        colors = ['red', 'green', 'blue', 'purple', 'yellow', 'teal', 'orange']
        ax.scatter(self.transaction['recency'], self.transaction['frequency'], self.transaction['monetary'], c=label, s=50,
                   cmap=mat.colors.ListedColormap(colors))
        ax.set_xlabel('Recency', rotation=150)
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Money', rotation=60)
        plt.title('User Segments based on RFM Model')
        plt.savefig('figures/segment_scatter_plot.png')

    # function to label each user as high growth or not
    def Label_Segments(self, rows):
        if rows['segment'] == 'best_users':
            return 1
        elif rows['segment'] == 'big_spender':
            return 1
        else:
            return 0

    def save_user_group_bar_chart(self):
        growth_count = self.transaction.groupby('high_growth').count()
        plt.bar(growth_count.index.values, growth_count['user'])
        plt.xlabel('Low Growth                High Growth')
        plt.ylabel('Number of Users')
        plt.savefig('figures/user_group_bar_chart.png')


    def create_features(self):
        user_data = self.raw_data

        # convert time in datetime
        user_data['time'] = pd.to_datetime(user_data['time'])

        # convert amount into dollar
        user_data['amount_in_cents'] = user_data['amount_in_cents']
        today = datetime.now()
        user_data['year'] = self.raw_data['time'].apply(lambda x: x.year)
        user_data['month'] = self.raw_data['time'].apply(lambda x: x.month)
        user_data['day_since'] = self.raw_data['time'].apply(lambda x: (today - x).days)
        user_data = self.raw_data[['user', 'time', 'amount_in_cents', 'year', 'month', 'day_since']]
        dataset_year = self.raw_data.groupby('user')['year', 'amount_in_cents'].agg(
            {'year': ['min', 'max'], 'amount_in_cents': ['sum', 'mean', 'count', 'max'], 'day_since': ['max'],
             'time': lambda date: (today - date.max()).days}).reset_index()
        dataset_year.columns = ['user', 'since', 'till', 'monetary', 'mean', 'frequency', 'max', 'cust_dur', 'recency']

        # get transaction count per month
        dataset_mcount = user_data.groupby(['user', 'month']).size().reset_index()
        dataset_mcount.columns = ['user', 'month', 'count']
        dataset_mcount = dataset_mcount.pivot(index='user', columns='month', values='count')
        dataset_mcount.columns = ['m{:d}c'.format(col) for col in dataset_mcount.columns]
        dataset_mcount = dataset_mcount.reset_index()

        # get transaction Amt per month
        dataset_mamt = user_data.groupby(['user', 'month']).agg({'amount_in_cents': ['sum']}).reset_index()
        dataset_mamt.columns = ['user', 'month', 'amt']
        dataset_mamt = dataset_mamt.pivot(index='user', columns='month', values='amt')
        dataset_mamt.columns = ['m{:d}a'.format(col) for col in dataset_mamt.columns]
        dataset_mamt = dataset_mamt.reset_index()

        # Merge all into one data frame
        header = ['user', 'since', 'till', 'cust_dur', 'monetary', 'mean', 'frequency', 'max', 'recency', 'm1c', 'm2c',
                  'm3c',
                  'm4c', 'm5c', 'm6c', 'm7c', 'm8c', 'm9c', 'm10c', 'm11c', 'm12c', 'm1a', 'm2a', 'm3a',
                  'm4a', 'm5a', 'm6a', 'm7a', 'm8a', 'm9a', 'm10a', 'm11a', 'm12a']

        final_dataset = pd.DataFrame(index=range(0, len(dataset_year)), columns=header)
        final_dataset = final_dataset.fillna(dataset_year)
        final_dataset = final_dataset.fillna(dataset_mcount)
        final_dataset = final_dataset.fillna(dataset_mamt)
        final_dataset = final_dataset.fillna(0)

        # Now combine months data to season
        # Get seasonal count
        final_dataset['spring_count'] = final_dataset['m1c'] + final_dataset['m2c'] + final_dataset['m3c']
        final_dataset['summer_count'] = final_dataset['m4c'] + final_dataset['m5c'] + final_dataset['m6c']
        final_dataset['fall_count'] = final_dataset['m7c'] + final_dataset['m8c'] + final_dataset['m9c']
        final_dataset['winter_count'] = final_dataset['m10c'] + final_dataset['m11c'] + final_dataset['m12c']

        final_dataset['spring_amt'] = final_dataset['m1a'] + final_dataset['m2a'] + final_dataset['m3a']
        final_dataset['summer_amt'] = final_dataset['m4a'] + final_dataset['m5a'] + final_dataset['m6a']
        final_dataset['fall_amt'] = final_dataset['m7a'] + final_dataset['m8a'] + final_dataset['m9a']
        final_dataset['winter_amt'] = final_dataset['m10a'] + final_dataset['m11a'] + final_dataset['m12a']

        final_dataset = final_dataset[
            ['user', 'cust_dur', 'monetary', 'mean', 'frequency', 'max', 'recency', 'spring_count', 'summer_count',
             'fall_count', 'winter_count', 'spring_amt', 'summer_amt', 'fall_amt', 'winter_amt']]


        #for calculating sales growth between years
        user_year = user_data.groupby(['user', 'year'])['amount_in_cents'].sum().reset_index()
        user_year.columns = ['user', 'year', 'amount']
        user_year['sales_growth'] = user_year[
            'amount'].diff()  # (user_year['amount'].diff()/user_year['amount'].shift(1))*100
        user_year.loc[user_year.user != user_year.user.shift(1), 'sales_growth'] = 0
        user_year = user_year.groupby(['user'])['sales_growth'].sum().reset_index()

        # merge sales growth in the final dataset
        final_dataset = final_dataset.merge(user_year)

        return final_dataset