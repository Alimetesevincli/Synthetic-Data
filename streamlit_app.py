import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import streamlit as st
import sdmetrics
from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.multi_table import QualityReport
            
import streamlit_fonksiyonlar as stf
import fonksiyonlar as f



#######################################################################
def main():#########DEĞİŞTİ##########
    st.title("Sentetik Veri Üretimi, Profil Çıkartma ve Karşılaştırma")
    task = st.selectbox("Ne yapmak istiyorsunuz?",options=["", "Sentetik Veri Üretme", "Profil Çıkartma", "Veri Karşılaştırma","Test","Model Karşılaştırmaları"], key="s_box1")

    if task == "Sentetik Veri Üretme":
        real_data = stf.upload_data("Upload Real Data CSV")
        

        if real_data is not None:
            stf.display_data_overview(real_data, "Real Data")

            ############################################
            tabs_1 = st.tabs(["Upload JSON Column Types", "Enter Column Types Manually"])
            with tabs_1[0]:
                col_types_json = stf.upload_column_types()
            with tabs_1[1]:
                col_types_text = stf.text_input_column_types()
            col_types = col_types_json if col_types_json else col_types_text

            if col_types:
                st.session_state.col_types = col_types
                st.write("### Column Types:")
                st.json(col_types)
            
            model_file = st.file_uploader("Upload a trained model (pickle file)", type="pkl")
            if model_file is not None:
                model = stf.load_trained_model(model_file)
                st.session_state.trained_model = model
                st.success("Model loaded successfully")
                
            ################################################

            sample_size = st.slider("Number of data rows to train the synthetic production model", min_value=0,value=100,step=100,max_value=real_data.shape[0])

            col1, col2,col3 = st.columns([1,1,1])
            cc = col1.multiselect("Pick CC models", list(f.metrikler("cc",real_data).keys()),key="cc",default="chi-sq")
            cn = col2.multiselect("Pick CN models", list(f.metrikler("cn",real_data).keys()),key="cn",default="ANOVA")
            nn = col3.multiselect("Pick NN models", list(f.metrikler("nn",real_data).keys()),key="nn",default="Spearman")

            col4, col5,col6 = st.columns([1,1,1])

            model_choice = col4.multiselect("Choose the Synthetic Model",options=["Gaussian","Ctgan"], key="model_choice")
            sentetik_size = col5.number_input("Syntetic Records Size", min_value=0,value=10000)
            seed = col6.number_input("Seed number (optional)", min_value=0, step=1, value=42)

            if "synthetic_data" not in st.session_state:
                st.session_state.synthetic_data = None

            if "trained_model" not in st.session_state:
                st.session_state.trained_model = None

            if "col_types" not in st.session_state:
                st.session_state.col_types = None

            if st.button("Generate Synthetic Data"):

                st.cache_data.clear()
                if st.session_state.trained_model:
                    synthetic_data = stf.generate_synthetic_data(real_data, sentetik_size,sample_size, seed,model_choice,_trained_model=st.session_state.trained_model)
                    st.session_state.synthetic_data=synthetic_data

                    if st.session_state.col_types:######## 2222222
                        synthetic_data = stf.saved_model_json(synthetic_data,_col_types=st.session_state.col_types)
                        real_data = stf.saved_model_json(real_data,_col_types=st.session_state.col_types)

                elif st.session_state.col_types: ###### 222222
                    synthetic_data = stf.generate_synthetic_data(real_data, sentetik_size,sample_size, seed,model_choice,_col_types=st.session_state.col_types)
                    st.session_state.synthetic_data=synthetic_data
                else:
                    synthetic_data = stf.generate_synthetic_data(real_data, sentetik_size,sample_size, seed,model_choice)
                    st.session_state.synthetic_data=synthetic_data

            model_bytes = stf.save_model_for_download(st.session_state.trained_model)
            st.download_button(label="Download Trained Model", data=model_bytes, file_name="trained_model.pkl",
                                mime="application/octet-stream")
            

            tabs = st.tabs(["Profiling", "Comparison Results", "Test", "model comparison"])

            with tabs[0]:
                if real_data is not None and st.session_state.synthetic_data is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        stf.profil_çıkarma(real_data,st.session_state.cc,st.session_state.cn,st.session_state.nn)
                    with col2:
                        stf.profil_çıkarma(st.session_state.synthetic_data,st.session_state.cc,st.session_state.cn,st.session_state.nn)

            with tabs[1]:
                if real_data is not None and st.session_state.synthetic_data is not None:
                    stf.data_comparison_main(real_data,st.session_state.synthetic_data,st.session_state.cc,st.session_state.cn,st.session_state.nn)

            with tabs[2]:
                if real_data is not None and st.session_state.synthetic_data is not None:
                    stf.test(real_data,st.session_state.synthetic_data)
            with tabs[3]:
                if real_data is not None and st.session_state.synthetic_data is not None:
                    stf.model_comparison(real_data,st.session_state.synthetic_data)

            # with tabs[4]:
            #     if real_data is not None and st.session_state.synthetic_data is not None:
            #         num_cols,cat_cols,types=f.num_cat_cols(real_data)

            #         from table_evaluator import TableEvaluator

            #         for name,sentetik_data in st.session_state.synthetic_data.items():

            #             table_evaluator = TableEvaluator(real_data, sentetik_data, cat_cols=cat_cols)

            #             st.write(table_evaluator.evaluate(target_col='fraud'))




            

    elif task == "Profil Çıkartma":
        real_data = stf.upload_data("Upload Your CSV file")
        st.write("Customize the stats function:")
        cc = st.multiselect("Pick CC models", list(f.metrikler("cc",real_data).keys()),key="cc",default="chi-sq")
        cn = st.multiselect("Pick CN models", list(f.metrikler("cn",real_data).keys()),key="cn",default="ANOVA")
        nn = st.multiselect("Pick NN models", list(f.metrikler("nn",real_data).keys()),key="nn",default="Spearman")

        if real_data is not None:
            stf.profil_çıkarma(real_data,st.session_state.cc,st.session_state.cn,st.session_state.nn)
        
    elif task == "Veri Karşılaştırma":
        real_data = stf.upload_data("Upload Real CSV file")
        synthetic_data = stf.upload_data("Upload Synthetic CSV file")

        st.write("Customize the stats function:")
        cc = st.multiselect("Pick CC models", list(f.metrikler("cc",real_data).keys()),key="cc",default="chi-sq")
        cn = st.multiselect("Pick CN models", list(f.metrikler("cn",real_data).keys()),key="cn",default="ANOVA")
        nn = st.multiselect("Pick NN models", list(f.metrikler("nn",real_data).keys()),key="nn",default="Spearman")

        
        if real_data is not None and synthetic_data is not None:

            stf.data_comparison_main(real_data,st.session_state.synthetic_data,st.session_state.cc,st.session_state.cn,st.session_state.nn)



    elif task =="Test":
        real_data = stf.upload_data("Upload Real CSV file")
        synthetic_data = stf.upload_data("Upload Synthetic CSV file")
        
        if real_data is not None and synthetic_data is not None:
            stf.test(real_data,synthetic_data)
            

    elif task=="Model Karşılaştırmaları":
    
        real_data = stf.upload_data("Upload Real CSV file")
        ctgan_synthetic_data = stf.upload_data("Upload Synthetic CSV file")

        if real_data is not None and ctgan_synthetic_data is not None:
            stf.model_comparison(real_data,ctgan_synthetic_data)



            


if __name__ == "__main__":
    main()