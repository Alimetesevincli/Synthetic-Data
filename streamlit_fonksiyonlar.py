import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import streamlit as st

def save_model_for_download(model):#######EKLENDİ######
    model_bytes = io.BytesIO()
    f.save_model(model, model_bytes)
    model_bytes.seek(0)
    return model_bytes

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


# Function to display data overview
def display_data_overview(data, title="An"):
    st.header(f"{title} Overview")
    st.dataframe(data.head())

@st.cache_data
def load_trained_model(uploaded_file):########EKLENDİ########
    return f.load_model(uploaded_file)
@st.cache_data()
def generate_synthetic_data(real_data, num_records,seed,_trained_model=None):#####DEĞİŞTİ########

    return f.sentetik_prod(real_data, num_records,seed,_trained_model)


def display_comparison_results(comparison_results, mean_diff, median_diff):
    st.subheader("Comparison Results")
    st.dataframe(comparison_results)
    st.write(f"Mean Relative Difference: {mean_diff}")
    st.write(f"Median Relative Difference: {median_diff}")


def plot_distributions(real_data, synthetic_data, columns):
    st.subheader("Distribution Comparison")
    for col in columns:
        st.write(f"#### {col}")
        fig, ax = plt.subplots()
        sns.kdeplot(real_data[col], label='Real Data', ax=ax)
        sns.kdeplot(synthetic_data[col], label='Synthetic Data', ax=ax)
        ax.legend()
        st.pyplot(fig)

@st.cache_data
def size_graph(df,titl,selected_column):
    st.pyplot(f.size_graph(df,titl,selected_column))


@st.cache_data
def normal_dist(df,datasets,num_cols,max_value):
    st.pyplot(f.normal_dist(df,datasets,num_cols,max_value))

@st.cache_data
def cat_freq(df,synthetic_table,column_name,name):
    st.pyplot(f.cat_freq(df,synthetic_table,column_name,name))

@st.cache_data
def test_streamlit(data,sentetik_df,selected_column,add_sentetik_list=[200],test_size=1000,add_sentetik_type=0):

    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report


    ##Arranging Test data
    test_data=data.sample(test_size,random_state=6).reset_index(drop=True)


    X_te = test_data.drop([selected_column], axis=1)
    y_te = test_data[selected_column]
    X_train_te, X_test_te, y_train_te, y_test_te = train_test_split(X_te, y_te, stratify=y_te, test_size = 0.3, random_state = 46)

    liste=[]
    parameters=[]



    ##Arr Train data
    for num in add_sentetik_list:

        ###############################
        final_data=pd.concat([X_train_te,y_train_te],axis=1)


        for add_num in add_sentetik_type:

            eklenecek=sentetik_df.loc[sentetik_df[selected_column]==add_num].sample(num,random_state=25)
            final_data=pd.concat([final_data,eklenecek]).reset_index(drop=True)

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
        "Number of synthetic used:":num,
        **percentage_dict,
        "Precision":precision_score(y_test_te, y_pred),
        "Recall" : recall_score(y_test_te, y_pred),
        "F1 Score": f1_score(y_test_te, y_pred),
        })

    return pd.DataFrame(parameters)

############ KOMPAKT FONKSİYONLAR #####################################
@st.cache_data
def profil_çıkarma(data):

    display_data_overview(data)
    st.dataframe(f.do_stats(data))

@st.cache_data
def veri_karşılaştırma(real_data,synthetic_data):
    display_data_overview(real_data, "Real Data")
    display_data_overview(synthetic_data, "Synthetic Data")
    comparision_df,mean_diff,median_diff=f.difference_p_value(real_data,synthetic_data)
    display_comparison_results(comparision_df, mean_diff, median_diff)

@st.cache_data
def compare_specified(real_data,synthetic_data,selected_columns):

    real_data = real_data[selected_columns]
    synthetic_data = synthetic_data[selected_columns]

    st.write("### Real Data Statistics")    
    st.dataframe(f.do_stats(real_data))
    st.write("### Synthetic Data Statistics")
    st.dataframe(f.do_stats(synthetic_data))


    comparison_results, mean_diff, median_diff = f.difference_p_value(real_data, synthetic_data)
    display_comparison_results(comparison_results, mean_diff, median_diff)
    plot_distributions(real_data, synthetic_data, selected_columns)

def data_comparison_main(real_data,synthetic_data):
    veri_karşılaştırma(real_data,synthetic_data)
    selected_columns = st.multiselect("Select Columns to Compare", real_data.columns.tolist())

    if len(selected_columns)>=2:
        compare_specified(real_data,synthetic_data,selected_columns)

def test(real_data,synthetic_data):
    st.write("Test")
    selected_column = st.selectbox("Select Column to test", real_data.columns.tolist(),placeholder="Choose an option",index=None)
    st.session_state.selected_column=selected_column


    if selected_column is not None:

        test_size = st.slider("Skew your set?", min_value=0, max_value=real_data.shape[0], value=real_data.shape[0],step=100)
        size_graph(real_data.sample(test_size),"Real",selected_column)
        size_graph(synthetic_data,"Sentetik",selected_column)

        add_sentetik_type = st.multiselect("Pick which value to add", real_data[selected_column].value_counts().index.to_list())

        min_value = int(st.number_input("Enter how much rows to be added at minimum from sythetic to real "))

        max_value = int(st.number_input("Enter how much rows to be added at most from sythetic to real "))

        step = int(st.number_input("Enter step size", min_value=1))



        if st.button("Start test"):
            add_sentetik_list = list(range(min_value, max_value, step))
            test_df=test_streamlit(real_data,synthetic_data,selected_column=selected_column,add_sentetik_list=add_sentetik_list,test_size=test_size,add_sentetik_type=add_sentetik_type)
            st.dataframe(test_df)

def model_comparison(real_data,synthetic_data):

    second_synthetic_data = upload_data("Upload Second Synthetic CSV file")
    first_synthetic = st.text_input("Name of your 1st model:","First synthetic" )
    second_synthetic = st.text_input("Name of your 2nd model:","Second synthetic" )

    if second_synthetic_data is not None:
        num_cols,cat_cols,_=f.num_cat_cols(real_data)

        datasets={
                first_synthetic:synthetic_data,
                second_synthetic:second_synthetic_data
                }


        graph_types=["normal distribition","categorical frequences","column pair trends"]

        selected_graphs = st.selectbox("Select Column to test", graph_types,placeholder="Choose an option",index=None)

        if selected_graphs=="normal distribition":

            numerical_columns = st.multiselect("Select Column to Graph", num_cols)

            if len(numerical_columns) != 0:

                for num_col in numerical_columns:
                    max = st.slider("Skew your set?", min_value=real_data[num_col].min(), max_value=real_data[num_col].max(), value=real_data[num_col].max(),step=100.00)

                    normal_dist(real_data,datasets,num_col,max)


        elif selected_graphs=="categorical frequences":

            categorical_columns = st.multiselect("Select Column to Graph", cat_cols)

            if len(categorical_columns)!=0:

                for cat_col in categorical_columns:
                    st.write(cat_col)
                    for name,data in datasets.items():
                        cat_freq(real_data,data,cat_col,name)


        elif selected_graphs=="column pair trends":

            import sdmetrics
            from sdmetrics.reports.single_table import DiagnosticReport
            from sdmetrics.reports.multi_table import QualityReport
    
            _,_,metadata=f.num_cat_cols(real_data)

            report = QualityReport()
            for model in datasets:
                report.generate(real_data,model, metadata)
                fig = report.get_visualization(property_name='Column Pair Trends')
                st.pyplot(fig)

