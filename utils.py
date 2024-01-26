# utils libraries
import pandas as pd
import numpy as np
import time,datetime,glob,requests,os,json,cdsapi
from fitter import Fitter, get_common_distributions, get_distributions
import geopandas as gpd
import pygeohash as pgh
import xarray as xr
import re

# visualization libraries
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotnine import *
import mplcyberpunk # library to add glowing effects to plots
from yellowbrick.cluster import *  # for clustering diagnostics
from termcolor import colored # for output colors
from IPython.display import Markdown as md
from matplotlib import font_manager

# pre-processing
from scipy.stats import uniform
from sklearn.datasets import make_blobs
from sklearn.preprocessing import RobustScaler,MinMaxScaler,normalize,OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn.inspection import permutation_importance
from yellowbrick.classifier import ROCAUC,PrecisionRecallCurve
from scikitplot.metrics import *
import miceforest as mf # imputation package
from miceforest import * # imputation package
from tqdm import tqdm # show a progress meter for apply operations
tqdm.pandas() # initialize tqdm for pandas
# from pandarallel import pandarallel
# pandarallel.initialize() # initialize pandarallel
# import mapply # multi-core apply function for pandas

# ml models
from xgboost import XGBClassifier
from sklearn.cluster import KMeans,DBSCAN
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor,NearestNeighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor,RandomForestRegressor,RandomForestClassifier
from mango import Tuner # bayesian optimization package

# settings
from credentials import credentials
gstyle = 'https://raw.githubusercontent.com/carminemnc/utils/main/'
# plt.style.use('https://raw.githubusercontent.com/carminemnc/utils/main/dark-theme.mplstyle') # custom matplotib style
pd.set_option('display.max_columns', 200)


class Enrico:
    
    def extratree_bayesian_regressor(self, x, y):

        # split into train/test (80/20)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42)

        # parameters space
        param_space = {
            'max_depth': range(3, 10),
            'min_samples_split': range(int(0.01*x.shape[0]), int(0.1*x.shape[0])),
            'min_samples_leaf': range(int(0.01*x.shape[0]), int(0.1*x.shape[0])),
            'max_features': ["sqrt", "log2", "auto"]
        }

        # objective function on train/validation with cross-validation
        def objective(space):
            results = []
            for hyper_params in space:
                # model
                model = ExtraTreesRegressor(**hyper_params)
                # cross validation score on 5 folds
                result = cross_val_score(
                    model, x_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()
                results.append(result)
            return results

        # optimize, maximizing the negative mean squared error
        tuner = Tuner(param_space,
                    objective,
                    dict(num_iteration=80, initial_random=10)
                    )
        optimisation_results = tuner.maximize()

        best_objective = optimisation_results['best_objective']
        best_params = optimisation_results['best_params']

        # results on test-set
        model = ExtraTreesRegressor(**best_params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        

        test_results = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f'Best RMSE on train-set: {best_objective}')
        print(f'RMSE on test-set: {test_results}')
        print(f'Best Parameters: {best_params}')

        return x_train, x_test, y_train, y_test, best_objective, best_params, model
    
    def bayesian_xgbclass(x, y,test_size=0.2,n_iterations = 10):

        # split into train/test (default 80/20)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=42)

        # parameters space
        param_space = {
            "learning_rate": uniform(0, 1),
            "gamma": uniform(0, 5),
            # "max_depth": range(1,10),
            # "n_estimators": range(1,300),
            # "booster":['gbtree','gblinear','dart']
            }

        # objective function on train/validation with cross-validation
        def objective(space):
            results = []
            for hyper_params in space:
                # model
                model = XGBClassifier(**hyper_params)
                # cross validation score on 5 folds
                result = cross_val_score(
                    model, x_train, y_train, scoring='accuracy', cv=5).mean()
                results.append(result)
            return results

        # optimize, maximizing the accuracy
        tuner = Tuner(
                    param_space,
                    objective,
                    dict(num_iteration=n_iterations, initial_random=10)
                    )
        optimisation_results = tuner.maximize()

        best_objective = optimisation_results['best_objective']
        best_params = optimisation_results['best_params']

        # results on test-set
        model = XGBClassifier(**best_params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        print(colored('Accuracy on test set:',color='cyan',attrs=['bold']) + f' {round(accuracy_score(y_test,y_pred),4)}')

        return x_train, x_test, y_train, y_test, best_objective, best_params, optimisation_results, model
    
class Maestro:
    
    def normalization(self,dataframe,method):
        
        if method=='minmax':
            dataframe = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())
        if method=='mean':
            dataframe = (dataframe - dataframe.mean()) / dataframe.std()
        
        
        return dataframe
    
    def outliers(data, column_name, output):

        
        q25 = data[column_name].quantile(0.25)
        q75 = data[column_name].quantile(0.75)
        iqr = q75-q25
        cut_off = iqr*1.5
        lower, upper = q25-cut_off, q75+cut_off

        outliers = data[(data[column_name] < lower) |
                        (data[column_name] > upper)]
        r_outliers = data[(data[column_name] > lower) &
                        (data[column_name] < upper)]

        print(f'25th quantile: {q25} \n75h quantile: {q75} \nIQR: {iqr}\nCut-Off Threshold: {cut_off} \
            \nLower Bound: {lower}\nUpper Bound: {upper}\n# of outliers: {len(outliers)}\n% of outliers: {len(outliers)/len(data)}')

        fig, ax = plt.subplots(1, 2)

        sns.boxplot(data[column_name],
                    notch=True,
                    showcaps=False,
                    flierprops={"marker": "o"},
                    boxprops={"facecolor": (.4, .6, .8, .5)},
                    medianprops={"color": "coral"},
                    fliersize=5, ax=ax[0]).set(title='Outliers boxplot')

        sns.boxplot(r_outliers[column_name],
                    notch=True,
                    showcaps=False,
                    flierprops={"marker": "o"},
                    boxprops={"facecolor": (.4, .6, .8, .5)},
                    medianprops={"color": "coral"},
                    fliersize=5, ax=ax[1]).set(title='Cleaned series')
        
        fig.show()

        if output == 'create_feature':
            data[column_name + '_outliers'] = data[column_name].apply(
                lambda x: 'outlier' if (x < lower) | (x > upper) else np.nan)
        elif output == 'replace_with_na':
            data[column_name] = data[column_name].apply(
                lambda x: np.nan if (x < lower) | (x > upper) else x)
        elif output == 'drop_outliers':
            data = data[(data[column_name] > lower) & (data[column_name] < upper)]

        return data
    
    def timestamp_feature_extractor(self,data,column_name,opts=None):
        
        """
        This function extract features from a datetime column.
        
        Feature options can be provided as a list by `opts` argument.
        
        Options: ['datetime','year','month','quarter','month','day','weekday','weekend','week',
                  'hour','minute','seconds',
                  'week',
                  'sin_month','cos_month',
                  'sin_week','cos_week',
                  'sin_weekday','cos_weekday',
                  'sin_hour','cos_hour'
                  ]
        
        Special features:
            - `datetime` extracting date in `YYYY-MM-DD HH:00:00` format
            - `weekday` # extracting weekday (Monday:0)
            - `weekend` extracting binary response for weekend `0,1`
            - `sine` and `cosine` transformation for `month,week,weekday,hour`
        
        Parameters:
        -----------
        
        data: `pandas dataframe object`
            Pandas dataframe object.
        column_name: `str`
            The dataframe column's name on which extracting features.
        opts: `list`, default `None`
            An optional list to extract more features.
            Options allowed: ['minute','year','weekend','day','quarter']

        Returns:
        data: `pandas dataframe object`
            Pandas dataframe with additional features.
        """
        
        data[column_name] = pd.to_datetime(data[column_name]).dt.tz_localize(None) # converting to datetime
        
        # checking optional_arguments list
        if opts is not None:
            if opts: # if list is not empty
                optslist = [x.lower() for x in opts]
                if 'date' in optslist:
                    data[column_name + '_date'] = pd.to_datetime(data[column_name].dt.strftime('%Y-%m-%d')) # extracting YYYY-MM-DD 
                if 'datetime' in optslist:
                    data[column_name + '_datetime'] = pd.to_datetime(data[column_name].dt.strftime('%Y-%m-%d %H:00:00')) # extracting YYYY-MM-DD HH:00:00
                if 'year' in optslist:
                    data[column_name + '_year'] = data[column_name].dt.year.astype(np.int64) # extracting year
                if 'quarter' in optslist:
                    data[column_name + '_quarter'] = data[column_name].dt.quarter.astype(np.int64) # extracting quarter         
                if 'month' in optslist:
                    data[column_name + '_month'] = data[column_name].dt.month.astype(np.int64) # extracting month
                if 'day' in optslist:
                    data[column_name + '_day'] = data[column_name].dt.day.astype(np.int64) # extracting day
                if 'hour' in optslist:
                    data[column_name + '_hour'] = data[column_name].dt.strftime('%H').astype(np.int64) # extracting hour
                if 'minute' in optslist:
                    data[column_name + '_minute'] = data[column_name].dt.minute.astype(np.int64) # extracting minute 
                if 'seconds' in optslist:
                    data[column_name + '_seconds'] = data[column_name].dt.second.astype(np.int64) # extracting seconds
                if 'weekday' in optslist:
                    data[column_name + '_weekday'] = data[column_name].dt.dayofweek.astype(np.int64) # extracting weekday (Monday:0)
                if 'weekend' in optslist:
                    data[column_name + '_weekend_dummy'] = data[column_name].dt.dayofweek.astype(np.int64) \
                        .apply(lambda x: 1 if x in [5, 6] else 0) # extracting binary response for weekend {0,1}
                if 'week' in optslist:
                    data[column_name + '_week'] = data[column_name].dt.isocalendar().week.astype(np.int64)
                    
                ''' sine and cosine transformations'''
                if 'sin_month' in optslist:
                    data[column_name + '_sin_month'] = np.sin(2*np.pi*data[column_name].dt.month.astype(np.int64)/12)
                if 'cos_month' in optslist:
                    data[column_name + '_cos_month'] = np.cos(2*np.pi*data[column_name].dt.month.astype(np.int64)/12)
                if 'sin_week' in optslist:
                    data[column_name + '_sin_week'] = np.sin(2*np.pi*data[column_name].dt.isocalendar().week.astype(np.int64)/52)
                if 'cos_week' in optslist:
                    data[column_name + '_cos_week'] = np.cos(2*np.pi*data[column_name].dt.isocalendar().week.astype(np.int64)/52)
                if 'sin_weekday' in optslist:
                    data[column_name + '_sin_weekday'] = np.sin(2*np.pi*data[column_name].dt.dayofweek.astype(np.int64)/7)
                if 'cos_weekday' in optslist:
                    data[column_name + '_cos_weekday'] = np.sin(2*np.pi*data[column_name].dt.dayofweek.astype(np.int64)/7)
                if 'sin_hour' in optslist:
                    data[column_name + '_sin_hour'] = np.sin(2*np.pi*data[column_name].dt.strftime('%H').astype(np.int64)/24)
                if 'cos_hour' in optslist:
                    data[column_name + '_cos_hour'] = np.cos(2*np.pi*data[column_name].dt.strftime('%H').astype(np.int64)/24)
                
            else:
                print('The list of optional features that you provided is empty.')
        else:
            print('You\'ve not provided any options, try with some options.')
        
        return data
    
class Voyager:
    
    def copernicus_downloader(self,variables,years,months,days,hours,sub_region,download_path,file_name):
        
        c = cdsapi.Client(url=credentials['copernicus_url'],
                          key=credentials['copernicus_key'],
                          progress=True)
        
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variables,
                'year': years,
                'month': months,
                'day': days,
                'time': hours,
                'area': sub_region
            },
            f'{download_path}/{file_name}.nc')
        
        return
    
    def copernicus_to_dataframe(self,file_path):
        
        # read data
        xrarray_data = xr.open_dataset(file_path)
        # to dataframe
        data = xrarray_data.to_dataframe().reset_index()
        
        return data
    
class Leonardo:
    
    def binary_target_ratio_plot(self,data,column_name,target_zero_name,target_one_name,font_color='white',plot_title=None):
        
        """
        Plot function for a binary target column.
        
        Parameters:
        -----------
        data: `pandas dataframe object`
            Pandas dataframe object.
        column_name: `str`
            The dataframe column's name of the target binary column.
        target_zero_name: `str`
            Custom name for target "0" class.
        target_one_name: `str`
            Custom name for target "1" class.
        font_color: `str`
            Font color for plot's text.
        column_name: `str`
            Custom plot title.

        Returns:
            Styled horizontal bar chart.
        """
        
        # target ratio
        tratio = round(data[column_name].value_counts(normalize=True),4)*100

        # target "0" name
        tzeroname = target_zero_name
        # target "1" name
        tonename = target_one_name
        
        '''Plot'''
        fig, ax = plt.subplots(1,1,figsize=(6.5, 2),dpi=150)
        # barplots
        ax.barh(column_name,tratio[0],alpha=0.9)
        ax.barh(column_name,tratio[1],left=tratio[0])
        # plot settings
        ax.set_xlim([0,100])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
        
        # target "0"
        ax.annotate(f'{tratio[0]:.2f} %',xy=(10,column_name),va = 'center', ha='center',color=font_color)
        ax.annotate(tzeroname,xy=(10,-0.1),va = 'center', ha='center',fontsize=10,color=font_color)
        
        # target "1"
        ax.annotate(f'{tratio[1]:.2f} %',xy=(90,column_name),va = 'center', ha='center',color=font_color)
        ax.annotate(tonename,xy=(90,-0.1),va = 'center', ha='center',fontsize=10,color=font_color)
        
        if plot_title is None:
            pass
        else:
            plt.title(plot_title)
            
        # plot show
        plt.show()
        
        return
    
class Rita:
    
    def kmeans_diagnostic(self,data,clusters_range,mode='basic',theme='dark'):
        
        # theme definition
        if theme=='dark':
            bg = '#242728'
            lbl = '#eaeaea'
        else:
            bg = '#eaeaea'
            lbl = '#242728'
        
        # lists for basic mode
        sil = []
        elb = []
        
        ''' Basic mode'''
        if mode=='basic':
            
            fig, ax = plt.subplots(ncols=3,figsize=(15,5))
            fig.tight_layout(pad=2)
            fig.set_facecolor(bg)
            
            km = KMeans(init='k-means++',n_init=12,max_iter=100)
            # distorsion: mean sum of squared distances to centers
            elb = KElbowVisualizer(km,k=clusters_range,ax=ax[0],locate_elbow=True)
            elb.fit(data)
            ax[0].legend(loc='upper left')
            ax[0].set_title('Distortion score Elbow for KMeans Clustering')
            # silhouette: mean ratio of intra-cluster and nearest-cluster distance
            sil = KElbowVisualizer(km,k=clusters_range,metric='silhouette',locate_elbow=True,ax=ax[1])
            sil.fit(data)
            ax[1].legend(loc='upper left')
            ax[1].set_title('Silhouette score Elbow for KMeans Clustering')
            # calinski_harabasz: ratio of within to between cluster dispersion
            cal = KElbowVisualizer(km,k=clusters_range,metric='calinski_harabasz',locate_elbow=True,ax=ax[2])
            cal.fit(data)
            ax[2].legend(loc='upper left')
            ax[2].set_title('Calinkski Harabasz score Elbow for KMeans Clustering')
            
            fig.show()
            
        ''' Advanced mode'''   
        if mode=='advanced':
            
            for nclstr in clusters_range:
                
                # fitting kmeans
                km = KMeans(n_clusters=nclstr,init='k-means++',n_init=12,max_iter=100)
                # plot settings
                fig, ax = plt.subplots(ncols=2,figsize=(10,5))
                fig.tight_layout(pad=2)
                fig.set_facecolor(bg)
                fig.suptitle(
                    f'Clustering diagnostic on {nclstr} clusters \
                    \n\n ASS: {round(silhouette_score(data,km.fit_predict(data)),4)}', 
                    fontsize=12,
                    y=1.10,
                    weight='bold',
                    color=lbl)
                
                for axis in range(0,2):
                    ax[axis].grid(False)
                    ax[axis].set_facecolor(bg)
                    ax[axis].xaxis.label.set_color(lbl)
                    ax[axis].tick_params(axis='x', colors=lbl)
                    ax[axis].tick_params(axis='y', colors=lbl)
                    ax[axis].get_yaxis().set_visible(False)
                
                # silhouette coefficient plot
                silhouette = SilhouetteVisualizer(km, colors='yellowbrick',ax=ax[0])
                silhouette.fit(data)
                ax[0].set_title(f'Silhouette score for {nclstr} clusters',color=lbl)
                ax[0].legend(labelcolor=lbl,loc='lower left')
                
                # intercluster distance
                icd = InterclusterDistance(km,ax=ax[1],legend_loc='lower left')
                icd.fit(data)
                ax[1].get_xaxis().set_visible(False)
                
                # show
                fig.show()
                

        return
    
    
    