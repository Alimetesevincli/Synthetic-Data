import json
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.express as px
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from scipy.stats import chi2_contingency
from scipy.stats import spearmanr
from functools import partial
from scipy.stats import f_oneway
from collections import Counter
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler,LabelEncoder
import sdv
from sdmetrics.visualization import get_column_plot
import streamlit as st
from sdv.single_table import GaussianCopulaSynthesizer,CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

#############################################################
##Type'ları bulan fonksiyon opsiyonel
def column_type(df):

    types={}

    for column_name in df.columns:

        dtype = len(list(df[column_name].unique()))

        if dtype>len(df)*0.0005:
            types.update({column_name: 'N'})
        else:
            types.update({column_name: 'C'})

    return types

def column_dict(df,cat_cols,num_cols):

    types={}

    for numerical_name in num_cols:
        types.update({numerical_name: 'N'})

    for categorical_name in cat_cols:
        types.update({categorical_name: 'C'})
    
    return types

def num_cat_cols(data):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    num_cols=[]
    cat_cols=[]
    meta={}

    for column in metadata.columns:
        if metadata.columns[column]["sdtype"]=="numerical":
            num_cols.append(column)
            meta.update({column:"N"})

        elif metadata.columns[column]["sdtype"]=="categorical":
            cat_cols.append(column)
            meta.update({column:"C"})

    return num_cols,cat_cols,meta



############################################################
##İstatisk fonksiyonları    
def cramers_v(column_1, column_2):
    tablo = pd.crosstab(column_1, column_2)
    chi2, _, _, _ = chi2_contingency(tablo)
    n = tablo.sum().sum()
    min_dim = min(tablo.shape) - 1
    v = np.sqrt(chi2 / (n * min_dim))
    return [v]

def pearson_corr(x, y):

    mean_x = x.mean()
    mean_y = y.mean()
    std_x = x.std()
    std_y = y.std()

    covariance = ((x - mean_x) * (y - mean_y)).mean()

    pearson_corr = covariance / (std_x * std_y)

    return [pearson_corr]

def spearman_correlation(x, y):
 
    correlation, p_value = spearmanr(x, y)
    return [correlation, p_value]

def anova_test(group_col, value_col,data):

    
    groups = group_col.unique()

    values=[value_col.iloc[list(data[group_col == group].index)] for group in groups]

    
    f_statistic, p_value = f_oneway(*values)
    return [f_statistic,p_value]

def chi_square_test(column1, column2):


    # Çapraz tablo (contingency table) oluşturma
    contingency_table = pd.crosstab(column1, column2)

    # Chi-square testini uygulama
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    return [chi2,p]

def point_biserial(column1, column2):

    from scipy.stats import pointbiserialr

    corr, p = pointbiserialr(column1, column2)

    return [corr,p]
    
#########################################################################################
#######Normalizasyon yöntemleri

def standart(df, columns):

    temp_list=pd.concat([df[columns[0]],df[columns[1]]],ignore_index=True)

    mean_value = temp_list.mean()
    std_value = temp_list.std()
        

    for column_name in columns:


        # Normalizasyon işlemi
        df.loc[:,column_name+'_Normalized'] =(df[column_name] - mean_value) / std_value
   
    
    return df

def min_max_normalizasyon(df, columns):

    temp_list=pd.concat([df[columns[0]],df[columns[1]]],ignore_index=True)

    min_value = temp_list.min()
    max_value = temp_list.max()
            

    for column_name in columns:

        df.loc[:,column_name+'_Normalized'] = (df[column_name] - min_value) / (max_value - min_value)


        
    return df

def box_cox_transformation(df, columns, small_value=1e-9):

    from scipy import stats

    for column_name in columns:

        df[column_name + '_BoxCox'], _ = stats.boxcox(df[column_name]+small_value)

    return df

def log_transformation(df, columns, small_value=1e-9):


    for column_name in columns:
        df[column_name + '_Log'] = np.log(df[column_name] + small_value)
    
    return df

def winsorize(df,columns, lower_percentile=0.25 ,upper_percentile=0.75):

    temp_list=pd.concat([df[columns[0]],df[columns[1]]],ignore_index=True)


    lower_bound = temp_list.quantile(lower_percentile)
    upper_bound = temp_list.quantile(upper_percentile)


    for column_name in columns:

        df.loc[:,column_name+'_Clipped']=df[column_name].clip(lower=lower_bound, upper=upper_bound)


    return df

def clip_values(df,columns, lower_bound=-5, upper_bound=5):

    for column_name in columns:

        df.loc[:,column_name+'_Clipped']=df[column_name].clip(lower=lower_bound, upper=upper_bound)
    return df

def robust(df,columns):

    temp_list=pd.concat([df[columns[0]],df[columns[1]]],ignore_index=True)

    rs=RobustScaler()
    

    for column_name in columns:

        tempdf=pd.DataFrame()
        tempdf[column_name]=temp_list
        rs.fit(tempdf)
    
        df.loc[:,column_name+'_Robust']=rs.transform(df[[column_name]])

    return df

#########################################################
## Models
def models(first,second,model,method,evaluation,df,typ):
    
    if len(model[method](df[first],df[second]))==1:
        value=model[method](df[first],df[second])[0]
        p_value=None
    else:
        value,p_value =model[method](df[first],df[second])
                    
    evaluation.append({ 
        "First Column":first,
        "Second Column" : second,
        "Type": typ,
        "Model": method,
        "Value": value,
        "P_value":p_value
                
        })

def metrikler(model_type,df):
    NN_models={"Spearman":spearman_correlation,"Pearson Corr":pearson_corr}
    CC_models={ "chi-sq":chi_square_test,"Cramers V":cramers_v}
    
    anova_with_df = partial(anova_test, data=df)

    CN_models={"ANOVA":anova_with_df,"Point Biserial":point_biserial}

    if model_type=="nn":
        return NN_models
    elif model_type=="cn":
        return CN_models
    elif model_type=="cc":
        return CC_models

#################################################################
######Ana fonksiyonlar
def do_stats(df,cat_cols=[],num_cols=[]):

    df=df.dropna(axis=0).reset_index(drop=True)

    typCC="C-C"
    typCN="C-N"
    typNN="N-N"
    CC_models = {name_metrik: value_metrik for name_metrik, value_metrik in metrikler("cc", df).items() if name_metrik in st.session_state.cc}
    CN_models = {name_metrik: value_metrik for name_metrik, value_metrik in metrikler("cn", df).items() if name_metrik in st.session_state.cn}
    NN_models = {name_metrik: value_metrik for name_metrik, value_metrik in metrikler("nn", df).items() if name_metrik in st.session_state.nn}

 

    if len(cat_cols)==0 and len(num_cols)==0:
        _,_,types=num_cat_cols(df)
    else:
        types=column_dict(df, cat_cols,num_cols)

    df=label_encode(df)
    df=df[list(types.keys())]

    evaluation=[]
    control_duplicate=[]

    for first in types.keys():
        for second in types.keys():

            if first == second or {first,second} in control_duplicate:
                continue

            if types[first] == "C" and types[second] == "C":
                
                control_duplicate.append({first,second})

                for method in CC_models.keys():

                    models(first,second,CC_models,method,evaluation,df,typCC)
                    

            elif types[first] == "C" and types[second] == "N":

                control_duplicate.append({first,second})

                for method in CN_models.keys():

                    models(first,second,CN_models,method,evaluation,df,typCN)
                
                
            elif types[first] == "N" and types[second] == "N":

                control_duplicate.append({first,second})

                for method in NN_models.keys():

                    models(first,second,NN_models,method,evaluation,df,typNN)


    result=pd.DataFrame(evaluation)
    
    return result

def difference(real_stats,synthetic_stats,methods=[winsorize,robust,min_max_normalizasyon]):

    difference_frame=pd.concat([real_stats[["First Column","Second Column","Type","Model","Value"]],synthetic_stats["Value"]],axis=1)

    difference_frame.columns=["First Column","Second Column","Type","Model","Real stats","Synthetic stats"]

    result_frame=pd.DataFrame()
    final_frame=pd.DataFrame()
    
    for method in methods:
        difference_frame=calceach(difference_frame,"Model",method)

    difference_frame["Relative Difference"]=abs(difference_frame.iloc[:,-2:-1].values-difference_frame.iloc[:,-1:].values)*100
    return difference_frame, difference_frame["Relative Difference"].mean(),difference_frame["Relative Difference"].median()

def difference_p_value(real_stats,sentetik_stats):

    df_p_value=distinct(real_stats)
    sentetik_p_value=distinct(sentetik_stats)


    df_p_value["Sentetik Value"]=sentetik_p_value["Value"]
    df_p_value["Sentetik P_value"]=sentetik_p_value["P_value"]
    
    df_p_value["Relative Difference"]=abs(df_p_value["P_value"]-df_p_value["Sentetik P_value"])*100

    return df_p_value,round(df_p_value["Relative Difference"].mean(),2),round(df_p_value["Relative Difference"].median(),2)

def test(data,models,sentetik_size,add_sentetik_list,test_size,add_sentetik_type):
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split

    ##Arranging Test data
    test_data=data.sample(test_size,random_state=6).reset_index(drop=True)
    size_graph(test_data,"Gerçek")

    X_te = test_data.drop(data.columns[-1], axis=1)
    y_te = test_data[data.columns[-1]]
    X_train_te, X_test_te, y_train_te, y_test_te = train_test_split(X_te, y_te, stratify=y_te, test_size = 0.3, random_state = 46)

    liste=[]
    parameters=[]
    
     #Arranging Sentetik
    for sentetik_model in models:

        sentetik_df=sentetik_model.sample(sentetik_size)
        size_graph(sentetik_df,"Sentetik")
    
        ##Arr Train data
        for num in add_sentetik_list:

            ###############################
            final_data=pd.concat([X_train_te,y_train_te],axis=1)


            for add_num in add_sentetik_type:
                
                eklenecek=sentetik_df.loc[sentetik_df[df.columns[-1]]==add_num].sample(num,random_state=25)
                final_data=pd.concat([final_data,eklenecek]).reset_index(drop=True)

            #size_graph(final_data,f"Sentetik Data- plus {num}")

            X_final_tr= final_data.drop(data.columns[-1], axis=1)
            y_final_tr= final_data[data.columns[-1]]
            #############################


            logistic_model = LogisticRegression(max_iter=5000)
            logistic_model.fit(X_final_tr, y_final_tr)

            y_pred =  logistic_model.predict(X_test_te)

            #print(classification_report(y_test_te, y_pred))

            values_final=list(final_data[final_data.columns[-1]].value_counts(sort=False))

            total_values = sum(values_final)
            
            percentage_dict = {f"Percentage of {val}": (values_final[enum] / total_values) * 100 
                   for enum, val in enumerate(final_data[final_data.columns[-1]].value_counts(sort=False).index)}


            ##EĞER BİNARY TAHMİN ETMİYORSAN WEIGHTED HESAPLANMALI BUNU ÇÖZ
            parameters.append({ 
            "Model":sentetik_model, 
            "Number of synthetic used:":num,
            **percentage_dict,            
            "Precision":precision_score(y_test_te, y_pred,),
            "Recall" : recall_score(y_test_te, y_pred,),
            "F1 Score": f1_score(y_test_te, y_pred,),
            })
        
        
    return pd.DataFrame(parameters)



def sentetik_prod(df,size,sample_size,seed,_col_types=None,_trained_model=None):#####DEĞİŞTİ#######
    
    model_dict={}

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    if _trained_model:
        synthetic_data = _trained_model.sample(size)
        model_dict.update({"Uploaded Model":synthetic_data.sample(size)})

    if "Ctgan" in st.session_state.model_choice:
        ctgan = CTGANSynthesizer(metadata)
        ctgan.fit(df.sample(sample_size,random_state=seed))
        model_dict.update({"Ctgan":ctgan.sample(size)})

    if "Gaussian" in st.session_state.model_choice:
        gaussian=GaussianCopulaSynthesizer(metadata)
        gaussian.fit(df.sample(sample_size,random_state=seed))
        model_dict.update({"Gaussian":gaussian.sample(size)})
    
    return model_dict
    
def Model_Evaluation(X_train,y_train,X_test,y_test):

    from sklearn.metrics import classification_report,f1_score
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    classifiers = {
        "Support Vector Machine":SVC(kernel="linear", C=0.025),
        "Random Forest":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        }
      
    for modelname, clf in classifiers.items():
        training=clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        st.write(green(modelname,'bold'))
        st.write(classification_report(y_test, y_pred))
        st.write(red('f1score',['bold','underlined']),f1_score(y_test, clf.predict(X_test), average='macro'))


###################################################
############yardımcı fonksiyonlar
def calceach(df,mod_column,normalization):

    result_frame=pd.DataFrame()

    for mod in list(df[mod_column].unique()):
        
        temp=normalization(df[df[mod_column]==mod],df.iloc[:,-2:].columns.to_list())

        result_frame=pd.concat([result_frame,temp])

    return result_frame

def label_encode(data):


    le = LabelEncoder()

    string_columns=[]

    for columns in data.columns:
        if isinstance(data[columns].iloc[0], str):
            string_columns.append(columns)

    for column in string_columns:
        data[column] = le.fit_transform(data[column])

    return data

def remove_outliers_iqr(df,features):
    from collections import Counter

    outlier_list = []
    
    for column in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[column], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[column],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determining a list of indices of outliers
        outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step )].index
        # appending the list of outliers 
        outlier_list.extend(outlier_list_column)
        
    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)        
    multiple_outliers = list( k for k, v in outlier_list.items() if v >=1 )
    
    # Calculate the number of records below and above lower and above bound value respectively
    out1 = df[df[column] < Q1 - outlier_step]
    out2 = df[df[column] > Q3 + outlier_step]
    
    print('Total number of deleted outliers is:', out1.shape[0]+out2.shape[0])
    df_out = df.drop(multiple_outliers, axis = 0).reset_index(drop=True)
    
    return df_out

def load_json(file_obj):#######EKLENDİ#######

    return json.load(file_obj)


def validate_dataset_columns(dataset: pd.DataFrame, columns_info: dict):#####EKLENDİ#######

    for col, col_type in columns_info.items():
        if col not in dataset.columns:
            raise ValueError(f"Column '{col}' is missing from the dataset")
        if str(dataset[col].dtype) != col_type:
            raise ValueError(f"Column '{col}' expected type '{col_type}' but got '{dataset[col].dtype}'")

def save_model(model, file_obj):######EKLENDİ#######

    pickle.dump(model, file_obj)


def load_model(file_obj):#######EKLENDİ######

    return pickle.load(file_obj)

def distinct(df):
    df.loc[df["P_value"]<0.05,"P_value"]=0
    df.loc[df["P_value"]>0.05,"P_value"]=1

    return df


########GRAPHS#########################33

def size_graph(df,titl,selected_column):
    target_column = selected_column
    
    value_counts = df[target_column].value_counts(sort=True)
    labels = value_counts.index
    values = value_counts.values
    
    percentages = (values / values.sum()) * 100
    
    n_colors = len(labels)
    palette = plt.cm.get_cmap('Set3')(np.linspace(0, 1, n_colors))
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    ax1.pie(percentages, labels=labels, autopct='%1.1f%%', 
            startangle=90, colors=palette,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
    ax1.set_title(f"Distribution of {target_column} in dataset (%)")
    
    sns.countplot(data=df, x=target_column, palette=palette, ax=ax2, order=labels)
    for i in ax2.containers:
        ax2.bar_label(i)
    ax2.set_title(titl)
    ax2.set_xlabel(target_column)
    
    plt.tight_layout()
    return fig


def cat_freq_plotly(data, sentetik_json, column_name):
    count1 = data[column_name].value_counts(normalize=True).reset_index()
    count1.columns = ['result', 'percentage']
    count1['dataset'] = 'Real Data'

    combined_counts = count1.copy()
    st.subheader(column_name)
    for name, sentetik in sentetik_json.items():
        count = sentetik[column_name].value_counts(normalize=True).reset_index()
        count.columns = ['result', 'percentage']
        count["dataset"] = name + " Data"
        combined_counts = pd.concat([combined_counts, count], ignore_index=True)

    fig = px.bar(combined_counts, x='result', y='percentage', color='dataset', barmode='group',
                 title='Categorical Percentage Frequency Comparison',
                 labels={'result': 'Result', 'percentage': 'Percentage'})

    return fig



def cat_freq_sdv(data,sentetik,column_name):
    from sdmetrics.visualization import get_column_plot

    fig = get_column_plot(
        real_data=data,
        synthetic_data=sentetik,
        column_name=column_name,
        plot_type='bar'
    )
    
    return fig


def normal_dist_plotly(df, datasets, column, max_value):
    df = df[df[column] <= max_value]
    for name, data in datasets.items():
        datasets[name] = data[data[column] <= max_value]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df[column], 
        name=f'{column} + Orijinal', 
        histnorm='probability density', 
        opacity=0.6,
        marker=dict(color='blue')  
    ))

    colors = ['red', 'green', 'purple', 'orange', 'cyan']  
    for i, (name, data) in enumerate(datasets.items()):
        fig.add_trace(go.Histogram(
            x=data[column], 
            name=f'{column} + {name}', 
            histnorm='probability density', 
            opacity=0.6,
            marker=dict(color=colors[i % len(colors)]) 
        ))

    fig.update_layout(
        title=f'Normal Dağılım Grafiği - {column}',
        xaxis_title=column,
        yaxis_title='Density',
        barmode='overlay',
        bargap=0.1,
        height=600,
        width=800,
        legend=dict(
            title='Veri Kümesi',
            x=0.8,
            y=0.9
        )
    )

    return fig
