import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import spacy
from spacy import displacy
import networkx as nx
import community as community_louvain
# from community import best_partition
import re
from adjustText import adjust_text
from PIL import Image
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='AFFIX PRODUCT EXPLORER',page_icon=':bar_chart:',layout='wide')
@st.cache
def segment_df():
    option = st.sidebar.selectbox("Select inputs", ('industry', 'segments', 'code'))
    df_seg=pd.read_csv('affixcon_segments.csv',encoding='latin-1',usecols=["industry","code","category","segment_name"])
    df_seg.category = df_seg.category.str.upper()
    df_seg.segment_name = df_seg.segment_name.str.title()
    df_seg = df_seg[df_seg['category'] != 'APP CATEGORY']

    col1,col2,col3=st.columns((3))

    with col1:
        if option=='industry':
            industry_list = df_seg['industry'].dropna().unique().tolist()
            selected_industry = st.selectbox(' :bookmark_tabs: Enter Industry:', industry_list)

            segment_industry_dict = df_seg.groupby('industry')['segment_name'].apply(list).to_dict()
            item_list = segment_industry_dict[selected_industry]
            select_all_segments = st.checkbox("Select All Segments")
            if select_all_segments:
                selected_segments = item_list
            else:
                selected_segments = st.multiselect("Select one or more segments:", item_list)

        elif option == 'segments':
            segment_list=df_seg['segment_name'].dropna().unique().tolist()
            # st.subheader('give segments as comma separated values')
            selected_segments = st.multiselect(" :bookmark_tabs: Enter segments as comma separated values",segment_list)
            if len(selected_segments)>0:
                selected_segments=[selected_segments]

        elif option == 'code':
            # st.subheader('give a code')
            selected_code = st.text_input(":bookmark_tabs: Enter code")
            item_list = []
            segment_industry_dict = df_seg.groupby('code')['segment_name'].apply(list).to_dict()
            def find_similar_codes(input_code, df):
                similar_codes = []
                for index, row in df.iterrows():
                    code = row['code']
                    if isinstance(code, str) and code.startswith(input_code):
                        similar_codes.append(code)
                return similar_codes
            

            user_contain_list = list(set(find_similar_codes(selected_code, df_seg)))

            if selected_code in user_contain_list:
                for code in user_contain_list:
                    item_list_code = segment_industry_dict[code]
                    for item in item_list_code:
                        item_list.append(item)
            else:
                item_list = []
            # Create a checkbox to select all segments
            select_all_segments = st.checkbox("Select All Segments")

            # If the "Select All Segments" checkbox is checked, select all segments
            if select_all_segments:
                selected_segments = item_list
            else:
                # Create a multiselect widget
                selected_segments = st.multiselect("Select one or more segments:", item_list)
        segment_category_dict = df_seg.set_index('segment_name')['category'].to_dict()
        result_dict = {}
        filtered_dict = {key: value for key, value in segment_category_dict.items() if key in selected_segments}

        for key, value in filtered_dict.items():

            if value not in result_dict:
                result_dict[value] = []

            result_dict[value].append(key)
            result_dict = {key: values for key, values in result_dict.items()}

        if 'BRAND VISITED' in result_dict and 'BRANDS VISITED' in result_dict:
            # Extend the 'a' values with 'a1' values
            result_dict['BRAND VISITED'].extend(result_dict['BRANDS VISITED'])
            # Delete the 'a1' key
            del result_dict['BRANDS VISITED']

        selected_category = st.sidebar.radio("Select one option:", list(result_dict.keys()))
        if selected_segments:
            if selected_category == 'INTERESTS':
                segment_list=result_dict['INTERESTS']
            elif selected_category == 'BRAND VISITED':
                segment_list=result_dict['BRAND VISITED']
            elif selected_category == 'PLACE CATEGORIES':
                segment_list=result_dict['PLACE CATEGORIES']
            elif selected_category == 'GEO BEHAVIOUR':
                segment_list=result_dict['GEO BEHAVIOUR']
        else:
            segment_list=[]

        
        # st.write(segment_list)
        for j in segment_list:
            st.sidebar.write(j)
    with col2:
        uploaded_file = st.file_uploader(" :file_folder: Upload a file", type=["Csv"])
    selected_columns = ['interests', 'brands_visited', 'place_categories', 'geobehaviour']
    def filter_condition(df,lst):
        filter_conditions = [df[col_name].apply(lambda x: any(item in str(x).split(',') for item in lst))
            for col_name in selected_columns]
        final_condition = filter_conditions[0]
        for condition in filter_conditions[1:]:
            final_condition = final_condition | condition
        df_new = df[final_condition]
        return df_new
    if uploaded_file is not None:
    # if uploaded_file>0:
        # df=pd.read_csv('..\sample data\multiple_contacts_new.csv')
        # Load data from the uploaded CSV 
        df = pd.read_csv(uploaded_file)
        with col3:
            st.write("Your uploaded data count is: ",len(df))
        df=filter_condition(df,selected_segments)
        with col3:
            st.write("Your Matched data count is: ",len(df))
        bin_edges = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,  float('inf')]  
        bin_labels = ['less than 20', 'Age 20-24', 'Age 25-29','Age 30-34', 'Age 35-39','Age 40-44', 'Age 45-49','Age 50-54', 'Age 55-59','Age 60-64', 'Age 65-69','Age 70-74', 'Age 75-79','Age 80-84','85+']  
    # 
        df['age_category'] = pd.cut(df['age'].fillna(-1), bins=bin_edges, labels=bin_labels, include_lowest=True)
        df['age_category'] = df['age_category'].cat.add_categories('unknown_age').fillna('unknown_age')
        gender_mapping = {'female': 'Female', 'male': 'Male'}
        df['gender'] = df['gender'].replace(gender_mapping)
        df['gender'].fillna('unknown_gender',inplace=True)
        df['income'].fillna('unknown_income',inplace=True)

        from sklearn.preprocessing import OneHotEncoder
        obj=OneHotEncoder()
        demographics_df=df[['income','gender','age_category']]
        def demographics(df):
            df=obj.fit_transform(df).toarray()
            encoded_feature_names = obj.get_feature_names_out(input_features=['income', 'gender', 'age_category'])
            df = pd.DataFrame(df, columns=encoded_feature_names)

            def remove_words_from_column_names(col_name):
                words_to_remove = ['income_', 'gender_', 'age_category_']
                for word in words_to_remove:
                    col_name = col_name.replace(word, '')
                return col_name

            df.rename(columns=remove_words_from_column_names, inplace=True)
            column_sums = df.sum()
            percentages = (column_sums / len(df)) * 100
            df = pd.DataFrame({'Column Name': df.columns, 'Percentage': percentages})
            return df
        
        demographics_df=demographics(demographics_df)
    return selected_segments,df_seg,result_dict
    
# selected_segments,df_seg,result_dict,col2,col3=segment_df()
#------------------------------------------------------------------------------------------------------------------------
# def file_uploader():
    # selected_segments,df_seg,result_dict,col21,col3=segment_df() 
    # with col21:
        # uploaded_file = st.file_uploader(" :file_folder: Upload a file", type=["Csv"])
    # return uploaded_file
    # selected_columns = ['interests', 'brands_visited', 'place_categories', 'geobehaviour']
    # def filter_condition(df,lst):
    #     filter_conditions = [df[col_name].apply(lambda x: any(item in str(x).split(',') for item in lst))
    #         for col_name in selected_columns]
    #     final_condition = filter_conditions[0]
    #     for condition in filter_conditions[1:]:
    #         final_condition = final_condition | condition
    #     df_new = df[final_condition]
    #     return df_new
    # if uploaded_file is not None:
    # # if uploaded_file>0:
    #     # df=pd.read_csv('..\sample data\multiple_contacts_new.csv')
    #     # Load data from the uploaded CSV 
    #     df = pd.read_csv(uploaded_file)
    #     with col3:
    #         st.write("Your uploaded data count is: ",len(df))
    #     df=filter_condition(df,selected_segments)
    #     with col3:
    #         st.write("Your Matched data count is: ",len(df))
    #     bin_edges = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,  float('inf')]  
    #     bin_labels = ['less than 20', 'Age 20-24', 'Age 25-29','Age 30-34', 'Age 35-39','Age 40-44', 'Age 45-49','Age 50-54', 'Age 55-59','Age 60-64', 'Age 65-69','Age 70-74', 'Age 75-79','Age 80-84','85+']  
    # # 
    #     df['age_category'] = pd.cut(df['age'].fillna(-1), bins=bin_edges, labels=bin_labels, include_lowest=True)
    #     df['age_category'] = df['age_category'].cat.add_categories('unknown_age').fillna('unknown_age')
    #     gender_mapping = {'female': 'Female', 'male': 'Male'}
    #     df['gender'] = df['gender'].replace(gender_mapping)
    #     df['gender'].fillna('unknown_gender',inplace=True)
    #     df['income'].fillna('unknown_income',inplace=True)

    #     from sklearn.preprocessing import OneHotEncoder
    #     obj=OneHotEncoder()
    #     demographics_df=df[['income','gender','age_category']]
    #     def demographics(df):
    #         df=obj.fit_transform(df).toarray()
    #         encoded_feature_names = obj.get_feature_names_out(input_features=['income', 'gender', 'age_category'])
    #         df = pd.DataFrame(df, columns=encoded_feature_names)

    #         def remove_words_from_column_names(col_name):
    #             words_to_remove = ['income_', 'gender_', 'age_category_']
    #             for word in words_to_remove:
    #                 col_name = col_name.replace(word, '')
    #             return col_name

    #         df.rename(columns=remove_words_from_column_names, inplace=True)
    #         column_sums = df.sum()
    #         percentages = (column_sums / len(df)) * 100
    #         df = pd.DataFrame({'Column Name': df.columns, 'Percentage': percentages})
    #         return df
        
    #     demographics_df=demographics(demographics_df)
    # return demographics_df
    






def main():
    st.title(" :bar_chart: AFFIX PRODUCT EXPLORER")
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
    # selected_segments,df_seg,result_dict=segment_df()
    # st.write(selected_segments)
    segment_df()
    # file_uploader()



if __name__ == "__main__":
    main()