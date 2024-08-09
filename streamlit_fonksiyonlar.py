import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import json
import fonksiyonlar as f
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import sdmetrics
from sdmetrics.visualization import get_column_plot
from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
import streamlit as st
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

###################upload,save vs.
def save_model_for_download(model):#######EKLENDİ######
    model_bytes = io.BytesIO()
    f.save_model(model, model_bytes)
    model_bytes.seek(0)
    return model_bytes

def saved_model_json(df,_col_types=None): #######EKLENDİ 2222222#####
    return f.preprocess_data(df,_col_types)    

def upload_data(prompt):
    data_file = st.file_uploader(prompt, type="csv")
    if data_file:
        return pd.read_csv(data_file)
    else:
        return None

def upload_column_types():#########EKLENDİ######
    col_types_file = st.file_uploader("Upload Column Types JSON", type="json")
    if col_types_file:
        return json.load(col_types_file)
    else:
        return None

def text_input_column_types(): ###########EKLENDİ 222222
    col_types_text = st.text_area("Enter Column Types JSON")
    if col_types_text:
        try:
            return json.loads(col_types_text)
        except json.JSONDecodeError:
            st.error("Invalid JSON format")
            return None
    else:
        return None

@st.cache_data
def load_trained_model(uploaded_file):########EKLENDİ########
    return f.load_model(uploaded_file)

@st.cache_data()
def generate_synthetic_data(real_data, num_records,sample_size,seed,model_choice,_col_types=None,_trained_model=None):#####DEĞİŞTİ########

    return f.sentetik_prod(df=real_data,size=num_records,sample_size=sample_size,seed=seed,_col_types=_col_types,_trained_model=_trained_model)


########################Display
@st.cache_data
def display_data_overview(data, title="An"):
    st.header(f"{title} Overview")
    st.dataframe(data.head())

@st.cache_data
def display_comparison_results(comparison_results, mean_diff, median_diff):
    st.subheader("Comparison Results")
    st.dataframe(comparison_results)
    st.write(f"Mean Relative Difference: {mean_diff}")
    st.write(f"Median Relative Difference: {median_diff}")

def plot_distributions(real_data, synthetic_data, columns):

    num_cols,cat_cols,_=f.num_cat_cols(real_data)
    real_data_iqr=f.remove_outliers_iqr(real_data,num_cols)
    synthetic_data_iqr=f.remove_outliers_iqr(synthetic_data,num_cols)

    for col in columns:

        fig = go.Figure()
        
        # Real data histogram
        fig.add_trace(go.Histogram(
            x=real_data_iqr[col], 
            name='Real Data', 
            histnorm='probability density',
            opacity=0.5,  # Increased transparency
            marker=dict(color='blue', line=dict(width=2, color='black'))  # Black border for better visibility
        ))
        
        # Synthetic data histogram
        fig.add_trace(go.Histogram(
            x=synthetic_data_iqr[col], 
            name='Synthetic Data', 
            histnorm='probability density',
            opacity=0.5,  # Increased transparency
            marker=dict(color='orange', line=dict(width=2, color='black'))  # Black border for better visibility
        ))
        
        fig.update_layout(
            barmode='overlay',
            title_text=f'Distribution of {col}', 
            xaxis_title=col, 
            yaxis_title='Density',
            bargap=0.1
        )
        
        st.plotly_chart(fig)


@st.cache_data
def size_graph(df,titl,selected_column):
    st.pyplot(f.size_graph(df,titl,selected_column))

@st.cache_data
def normal_dist_plotly(df,datasets,num_cols,max_value):
    st.plotly_chart(f.normal_dist_plotly(df,datasets,num_cols,max_value))

def cat_freq_plotly(real_data,datasets,cat_col):
    st.plotly_chart(f.cat_freq_plotly(real_data,datasets,cat_col))

@st.cache_data
def cat_freq_sdv(df,synthetic_table,column_name):
    st.plotly_chart(f.cat_freq_sdv(df,synthetic_table,column_name))

@st.cache_data
def test_streamlit(data,sentetik_models,selected_column,add_sentetik_list=[200],test_size=1000,add_sentetik_type=0):

    ##Arranging Test data
    test_data=data.sample(test_size,random_state=6).reset_index(drop=True)


    X_te = test_data.drop([selected_column], axis=1)
    y_te = test_data[selected_column]
    X_train_te, X_test_te, y_train_te, y_test_te = train_test_split(X_te, y_te, stratify=y_te, test_size = 0.3, random_state = 46)

    liste=[]
    parameters=[]
    flag=0

    for name,sentetik_df in sentetik_models.items():
        for num in add_sentetik_list:

            ###############################
            final_data=pd.concat([X_train_te,y_train_te],axis=1)


            for add_num in add_sentetik_type:

                eklenecek=sentetik_df.loc[sentetik_df[selected_column]==add_num].sample(num,random_state=25)
                final_data=pd.concat([final_data,eklenecek]).reset_index(drop=True)

            if flag==0:
                st.subheader(f"# of {num} synthetic rows added ")
                size_graph(final_data,f"Sentetik Data- plus {num}",selected_column=selected_column)

            X_final_tr= final_data.drop([selected_column], axis=1)
            y_final_tr= final_data[selected_column]
            #############################


            logistic_model = LogisticRegression(max_iter=1000)
            logistic_model.fit(X_final_tr, y_final_tr)

            y_pred =  logistic_model.predict(X_test_te)

            #print(classification_report(y_test_te, y_pred))

            values_final=list(final_data[selected_column].value_counts(sort=False))

            total_values = sum(values_final)

            percentage_dict = {f"Percentage of {val}": (values_final[enum] / total_values) * 100
                    for enum, val in enumerate(final_data[selected_column].value_counts(sort=False).index)}


            ##EĞER BİNARY TAHMİN ETMİYORSAN WEIGHTED HESAPLANMALI BUNU ÇÖZ
            parameters.append({
            "Model": name,
            "Number of synthetic used:":num,
            **percentage_dict,
            "Precision":precision_score(y_test_te, y_pred),
            "Recall" : recall_score(y_test_te, y_pred),
            "F1 Score": f1_score(y_test_te, y_pred),
            })

        flag=1

    return pd.DataFrame(parameters)

######## Yeni
def model_ev(real_data,synthetic_data_dict):

    X = real_data.drop(real_data.columns[-1], axis=1)
    y = real_data[real_data.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.3, random_state = 46)
    st.subheader("Real Data")
    f.Model_Evaluation(X_train, y_train, X_test, y_test)

    for name, data in synthetic_data_dict.items():
        XS = data.drop(data.columns[-1], axis=1)
        YS = data[data.columns[-1]]
        st.subheader(name)
        f.Model_Evaluation(XS, YS, X_test, y_test)

def table_eval(real_data,synth_data,cat_cols):
    from table_evaluator import load_data, TableEvaluator
    evaluator = TableEvaluator(real_data, synth_data, cat_cols=cat_cols)
    st.pyplot(evaluator.visual_evaluation())


############ KOMPAKT FONKSİYONLAR #####################################
@st.cache_data
def profil_çıkarma(data,cc,cn,nn):

    # display_data_overview(data)
    if type(data)==dict:
        for name,data in data.items():
            st.subheader(name)
            st.dataframe(f.do_stats(data))
    else:
        st.subheader("Real Data")
        st.dataframe(f.do_stats(data))


@st.cache_data
def veri_karşılaştırma(real_data,synthetic_data,cc,cn,nn):

    # display_data_overview(real_data, "Real Data")
    # display_data_overview(synthetic_data, "Synthetic Data")

    real_stats=f.do_stats(real_data)
    synthetic_stats=f.do_stats(synthetic_data)

    if real_stats["P_value"].isnull().any():
        comparision_df,mean_diff,median_diff=f.difference(real_stats,synthetic_stats)
    else:
        comparision_df,mean_diff,median_diff=f.difference_p_value(real_stats,synthetic_stats)

    display_comparison_results(comparision_df, mean_diff, median_diff)

@st.cache_data
def compare_specified(real_data,synthetic_data,selected_columns,cc,cn,nn):

    real_data = real_data[selected_columns]
    synthetic_data = synthetic_data[selected_columns]

    real_stats=f.do_stats(real_data)
    synthetic_stats=f.do_stats(synthetic_data)

    if real_stats["P_value"].isnull().any():
        comparision_df,mean_diff,median_diff=f.difference(real_stats,synthetic_stats)
    else:
        comparision_df,mean_diff,median_diff=f.difference_p_value(real_stats,synthetic_stats)

    display_comparison_results(comparision_df,mean_diff,median_diff)

    plot_distributions(real_data, synthetic_data, selected_columns)

def data_comparison_main(real_data,synthetic_data_dict,cc,cn,nn):

    cols= st.columns(len(synthetic_data_dict))

    for num,(name,synthetic_data) in enumerate(synthetic_data_dict.items()):
        with cols[num]:
            st.subheader(name + "Data - Real Data")
            veri_karşılaştırma(real_data,synthetic_data,cc,cn,nn)

    with st.form("model_comparison_form"):
        selected_columns = st.multiselect("Select Columns to Compare", real_data.columns.tolist())
        st.form_submit_button("Submit")

    cols2 = st.columns(len(synthetic_data_dict))
    
    if len(selected_columns)>=2:
        for num,(name,synthetic_data) in enumerate(synthetic_data_dict.items()):
            with cols2[num]:
                st.subheader(name + "Data - Real Data")
                compare_specified(real_data,synthetic_data,selected_columns,cc,cn,nn)


        

def test(real_data,synthetic_data_dict):##update
    st.write("Test")
    selected_column = st.selectbox("Select Column to test", real_data.columns.tolist(),placeholder="Choose an option",index=None)
    #st.session_state.selected_column=selected_column


    if selected_column is not None:
        with st.form("test_form"):
            test_size = st.slider("Skew your set?", min_value=0, max_value=real_data.shape[0], value=real_data.shape[0],step=100)          
            add_sentetik_type = st.multiselect("Pick which value to add", real_data[selected_column].value_counts().index.to_list())
            min_value = int(st.number_input("Enter how much rows to be added at minimum from sythetic to real "))
            max_value = int(st.number_input("Enter how much rows to be added at most from sythetic to real "))
            step = int(st.number_input("Enter step size", min_value=1))
            st.form_submit_button('Submit picks')


        size_graph(real_data.sample(test_size),"Real",selected_column)

        for name,synthetic_data in synthetic_data_dict.items():
            size_graph(synthetic_data,name,selected_column)


        if st.button("Start test"):
            add_sentetik_list = list(range(min_value, max_value, step))
            test_df=test_streamlit(real_data,sentetik_models=synthetic_data_dict,selected_column=selected_column,add_sentetik_list=add_sentetik_list,test_size=test_size,add_sentetik_type=add_sentetik_type)
            st.dataframe(test_df)





def model_comparison(real_data,synthetic_data_dict):

    added_synthetic_data = upload_data("Upload Second Synthetic CSV file")

    graph_types=["normal distribition","categorical frequences","column pair trends"]
    num_cols,cat_cols,_=f.num_cat_cols(real_data)

    datasets={}

    for synth_name,synth_data in synthetic_data_dict.items():
        datasets.update({synth_name:synth_data})

    if added_synthetic_data is not None:
        

        with st.form("comparison_form"):
            added_synthetic = st.text_input("Name of your Added model's data:","Added synthetic" )
            selected_graphs = st.selectbox("Select Column to test", graph_types,placeholder="Choose an option",index=None)
            st.form_submit_button('Submit Comparison Picks')

        datasets.update({added_synthetic:added_synthetic_data})


        if selected_graphs=="normal distribition":
            
            real_data_iqr=f.remove_outliers_iqr(real_data, num_cols)
            datasets_iqr={}

            for name,data in datasets.items():
                data_iqr=f.remove_outliers_iqr(data, num_cols)
                datasets_iqr.update({name:data_iqr})

            numerical_columns = st.multiselect("Select Column to Graph", num_cols)
            if len(numerical_columns) != 0:

                for num_col in numerical_columns:
                    max = st.slider("Skew your set?", min_value=real_data_iqr[num_col].min(), max_value=real_data_iqr[num_col].max(), value=real_data_iqr[num_col].max(),step=real_data_iqr[num_col].max()/10)

                    normal_dist_plotly(real_data_iqr,datasets_iqr,num_col,max)


        elif selected_graphs=="categorical frequences":

            categorical_columns = st.multiselect("Select Column to Graph", cat_cols)

            if len(categorical_columns)!=0:

                for cat_col in categorical_columns:
                    cat_freq_plotly(real_data,datasets,cat_col)
             


        elif selected_graphs=="column pair trends":

            
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(real_data)
            report = QualityReport()

            for name,model in datasets.items():
                report.generate(real_data,model, metadata.to_dict())
                fig = report.get_visualization(property_name='Column Pair Trends')
                st.subheader(name)
                st.plotly_chart(fig)

