import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import streamlit as st
import sdmetrics
from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.multi_table import QualityReport
            

import streamlit_fonksiyonlar as sf
from streamlit_fonksiyonlar import model_comparison,test,data_comparison_main,upload_data,display_data_overview,upload_column_types,generate_synthetic_data,display_comparison_results,plot_distributions,size_graph,normal_dist,test_streamlit,profil_çıkarma,veri_karşılaştırma,compare_specified,cat_freq
import fonksiyonlar as f


#######################################################################
def main():#########DEĞİŞTİ##########
    st.title("Sentetik Veri Üretimi, Profil Çıkartma ve Karşılaştırma")
    task = st.selectbox("Ne yapmak istiyorsunuz?",options=["", "Sentetik Veri Üretme", "Profil Çıkartma", "Veri Karşılaştırma","Test","Model Karşılaştırmaları"], key="s_box1")

    if task == "Sentetik Veri Üretme":
        real_data = upload_data("Upload Real Data CSV")

        if real_data is not None:
            display_data_overview(real_data, "Real Data")
            col_types = upload_column_types()
            if col_types:
                real_data = real_data.astype(col_types)
                st.write("### Column Types:")
                st.json(col_types)


            model_file = st.file_uploader("Upload a trained model (pickle file)", type="pkl")
            if model_file is not None:
                model = sf.load_trained_model(model_file)
                st.session_state.trained_model = model
                st.success("Model loaded successfully")
                model_bytes = sf.save_model_for_download(st.session_state.trained_model)
                st.download_button(label="Download Trained Model", data=model_bytes, file_name="trained_model.pkl",
                                   mime="application/octet-stream")


            #cc = st.multiselect("Pick CC models", list(f.metrikler("cc",real_data).keys()))
            #cn = st.multiselect("Pick CN models", list(f.metrikler("cn",real_data).keys()))
            #nn = st.multiselect("Pick NN models", list(f.metrikler("nn",real_data).keys()))

            sample_size = st.number_input("Enter number of synthetic records to generate", min_value=1)
            seed = st.number_input("Enter seed number (optional)", min_value=0, step=1, value=42)

            if st.button("Generate Synthetic Data"):
                if "trained_model" in st.session_state:
                    synthetic_data = generate_synthetic_data(real_data, sample_size, seed, _trained_model=st.session_state.trained_model)
                else:
                    synthetic_data = generate_synthetic_data(real_data, sample_size, seed)

                if synthetic_data not in st.session_state:
                    st.session_state.synthetic_data=synthetic_data

                st.dataframe( st.session_state.synthetic_data)
                

            tabs = st.tabs(["Profiling", "Comparison Results", "Test", "model comparison"])

            with tabs[0]:
                if real_data is not None and st.session_state.synthetic_data is not None:
                    st.write("Gerçek Veri")
                    profil_çıkarma(real_data)
                    st.write("Sentetik Veri")
                    profil_çıkarma(st.session_state.synthetic_data)


            with tabs[1]:
                if real_data is not None and st.session_state.synthetic_data is not None:

                    data_comparison_main(real_data,st.session_state.synthetic_data)

            with tabs[2]:
                if real_data is not None and st.session_state.synthetic_data is not None:
                    test(real_data,st.session_state.synthetic_data)

            with tabs[3]:
                if real_data is not None and st.session_state.synthetic_data is not None:
                    model_comparison(real_data,st.session_state.synthetic_data)

            
            # if st.button("Profil çıkarma"):
            #     profil_çıkarma( st.session_state.synthetic_data)

            # if st.button("Veri Karşılaştırma"):
            #     veri_karşılaştırma(real_data,st.session_state.synthetic_data)


           

            

    elif task == "Profil Çıkartma":
        real_data = upload_data("Upload Your CSV file")
        if real_data is not None:
            profil_çıkarma(real_data)
        
    elif task == "Veri Karşılaştırma":
        real_data = upload_data("Upload Real CSV file")
        synthetic_data = upload_data("Upload Synthetic CSV file")



        if real_data is not None and synthetic_data is not None:

            data_comparison_main(real_data,synthetic_data)


    elif task =="Test":
        real_data = upload_data("Upload Real CSV file")
        synthetic_data = upload_data("Upload Synthetic CSV file")
        
        if real_data is not None and synthetic_data is not None:
            test(real_data,synthetic_data)
            

    elif task=="Model Karşılaştırmaları":
    
        real_data = upload_data("Upload Real CSV file")
        ctgan_synthetic_data = upload_data("Upload Synthetic CSV file")

        if real_data is not None and ctgan_synthetic_data is not None:
            model_comparison(real_data,ctgan_synthetic_data)



            


if __name__ == "__main__":
    main()