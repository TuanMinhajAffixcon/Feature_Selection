import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import spacy
from spacy import displacy
import networkx as nx
import community as community_louvain
from community import best_partition
import re
# from adjustText import adjust_text
# from PIL import Image
# import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
# from bs4 import BeautifulSoup

st.set_page_config(page_title='AFFIX PRODUCT EXPLORER',page_icon=':bar_chart:',layout='wide')

def main():
    st.title(" :bar_chart: AFFIX PRODUCT EXPLORER")
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
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

#---------------------------------------------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------------------------------------------------
    with col2:
        uploaded_file = st.file_uploader(" :file_folder: Upload a file", type=["csv",'txt','xlsx','xls'])
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
        # st.write(demographics_df)

        col1,col2=st.columns((0.75,0.25))
        def bar_chart(df):
            fig, ax = plt.subplots(figsize=(12, 5))
            bars = plt.bar(df['Column Name'], df['Percentage'], color='#0083B8')
            plt.xlabel('Demographics',color='white')
            plt.ylabel('Percentage (%)',color='white')
            plt.title('Percentage of Each Demographics',color='white')
            plt.xticks(rotation=90,color='white')
            plt.yticks(color='white')


            for bar, value in zip(bars, df['Percentage']):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', rotation=90,color='white')

            fig.patch.set_alpha(0.0)
            ax.set_facecolor('none')
            boundaries=['top','right','bottom','left']
            for boundary in boundaries:
                ax.spines[boundary].set_visible(False)
            return plt
            
        with col1:
            bar_plot = bar_chart(demographics_df)
            st.pyplot(bar_plot)
        with col2:
            select_all_segments=st.selectbox('Selection Demographics',('Select All Demographics','Custom Demographics'))
        
        if select_all_segments=='Select All Demographics':
            df_filtered=df
    
        else:
            with col2:
                selected_age_category = st.multiselect("Select Age Category", ('less than 20', 'Age 20-24', 'Age 25-29','Age 30-34', 'Age 35-39','Age 40-44', 'Age 45-49','Age 50-54', 'Age 55-59','Age 60-64', 'Age 65-69','Age 70-74', 'Age 75-79','Age 80-84','85+'))
                selected_gender_category = st.multiselect("Select Gender Category", ('Male','Female'))
                selected_income_category = st.multiselect("Select Income Category", ("Very Low (Under $20,799)", "Low ($20,800 - $41,599)", "Below Average ($41,600 - $64,999)","Average ($65,000 - $77,999)", "Above Average ($78,000 - $103,999)","High ($104,000 - $155,999)","Very high ($156,000+)"))

                filter_criteria=selected_age_category+selected_gender_category+selected_income_category
                df_filtered = df[df.isin(filter_criteria).any(axis=1)]

        with col3:
            st.write("Demographics matched data count is: ",len(df_filtered))
        # st.write(selected_segments)
#-----------------------------------------------------------------------------------------------------------------------------------
        with col2:
            st.subheader(' :chart_with_upwards_trend: Executable Methods')
            approach=st.radio("Choose one option:", ['Plot chart only','Relationship Graph only',
                        'Plot chart first and Relationship Graph after','Relationship Graph first and Plot Chart after'])
        
        def Plot_chart_method(df_filtered,selected_segments,result_dict):
            selected_columns = ['interests', 'brands_visited', 'place_categories', 'geobehaviour']
            segments_filtered_df=df_filtered[selected_columns]
            
            for item in selected_segments:
                segments_filtered_df[item] = segments_filtered_df.apply(lambda row: 1 if any(
                    str(item) in val.split(',') if isinstance(val, str) else False for val in row) else 0, axis=1)
            segments_filtered_df.drop(selected_columns,axis=1,inplace=True)

            result_dict = {key: [value for value in result_dict[key] if value in selected_segments] for key in result_dict}
            # Remove keys with empty lists
            result_dict = {key: values for key, values in result_dict.items() if values}
            
            new_order = ["BRAND VISITED", "PLACE CATEGORIES", "INTERESTS", "GEO BEHAVIOUR"]
            items = list(result_dict.keys())
            sorted_items = sorted(items, key=lambda x: new_order.index(x))



            new_columns_order=[]
            for item in sorted_items:
                new_columns_order.extend(result_dict[item])
            segments_filtered_df = segments_filtered_df[new_columns_order]

            prevalence = segments_filtered_df.agg(lambda col: round(col.mean() * 100, 2)).reset_index()
            prevalence.columns = ['Segments','prevalence']
            segments_filtered_df = segments_filtered_df.replace(0, 'no data')

            overlap=segments_filtered_df
            for cat in result_dict.keys():
                seg=result_dict[cat]
                sum_non_zero = overlap[seg].apply(lambda col: col[col == 1].sum(),axis=1)-1
                count=overlap[seg].count(axis=1)
                row_mean = (sum_non_zero/count)*100
                overlap[seg] = overlap[seg].mask(overlap[seg] == 1, row_mean, axis=0) 

            sum_non_data_column = overlap.apply(lambda col: col[col != 'no data' ].sum(),axis=0)
            count_non_data_column=overlap.apply(lambda col: col[col != 'no data' ].count(),axis=0)
            column_mean=round(sum_non_data_column/count_non_data_column,2)
            overlap=pd.DataFrame([column_mean],columns=overlap.columns).fillna(0).transpose().reset_index()
            overlap.columns=['Segments','overlap']

            merged_op = pd.merge(prevalence, overlap,on='Segments',how='inner').set_index("Segments").transpose()
            for new_col, original_cols in result_dict.items():
                merged_op[new_col + '_avg'] = merged_op[original_cols].mean(axis=1)
                merged_op[new_col + '_std'] = merged_op[original_cols].std(axis=1,ddof=0)

            new_row_values = []
            for _, row in merged_op.iterrows():
                new_row = []
                for key, original_cols in result_dict.items():
                    new_row.extend((row[original_cols] - row[key + '_avg']) / row[key + '_std'])
                new_row_values.append(new_row)
            df_new_row = pd.DataFrame(new_row_values, columns=merged_op.columns[:-len(result_dict) * 2])

            merged_op = df_new_row
            merged_op = merged_op.fillna(0)
            merged_op.index = merged_op.index.map({0: 'prevalence', 1: 'overlap'})

            df_quadrant = merged_op.transpose()
            df_quadrant['quadrant'] = 'None'
            df_quadrant.loc[(df_quadrant['prevalence'] > 0) & (df_quadrant['overlap'] >= 0), 'quadrant'] = 'Top_right'
            df_quadrant.loc[(df_quadrant['prevalence'] <= 0) & (df_quadrant['overlap'] >= 0), 'quadrant'] = 'Top_left'
            df_quadrant.loc[(df_quadrant['prevalence'] <= 0) & (df_quadrant['overlap'] < 0), 'quadrant'] = 'Bottom_left'
            df_quadrant.loc[(df_quadrant['prevalence'] > 0) & (df_quadrant['overlap'] < 0), 'quadrant'] = 'Bottom_right'

            with col1:
                select_all_segments = st.checkbox("Select All Segments to show the Plot")
            def scatter_plot(df):
                fig, ax = plt.subplots(figsize=(12, 5))  
                scatter=plt.scatter(df.loc['prevalence'], df.loc['overlap'], color='blue', marker='o', label='All Segments')
                
                for x, y, word in zip(df.loc['prevalence'], df.loc['overlap'], df.columns):
                    plt.annotate(word, (x, y), textcoords="offset points",xytext=(10,-200), ha='center',color='white',
                                arrowprops=dict(facecolor='red',edgecolor='blue',arrowstyle='->',connectionstyle='angle'))

                # # Highlight x and y axes
                plt.axhline(0, color='red', linewidth=1, linestyle='--')  # Highlight x axis
                plt.axvline(0, color='red', linewidth=1, linestyle='--')  # Highlight y axis
                # plt.xlabel(color='white')
                fig.patch.set_alpha(0.0)
                ax.set_facecolor('none')
                boundaries=['top','right','bottom','left']
                for boundary in boundaries:
                    ax.spines[boundary].set_visible(False)
                return plt

            with col1:
                if select_all_segments:
                    scatter_plot(merged_op)
                else:
                    selected_keys = st.multiselect("Select Category:", list(result_dict.keys()))
                    # Create an empty list to store selected segments
                    selected_segments = []

                    # Iterate through selected keys and add corresponding segments to the selected_segments list
                    for key in selected_keys:
                        selected_segments.extend(result_dict[key])

                    merged_op=merged_op[selected_segments]
                    scatter_plot(merged_op)
                plt.title('Scatter Plot of Prevalence vs Overlap',color='white')
                plt.xlabel('Prevalence',color='white')
                plt.ylabel('Overlap',color='white')
                plt.legend()
                st.pyplot(plt.gcf())
        
                
#-----------------------------------------------------------------------------------------------------------------------------
                # Calculate the distance from the origin (0, 0) for each point
                distances = np.sqrt(df_quadrant['prevalence'] ** 2 + df_quadrant['overlap'] ** 2)
                # Calculate the distance from the origin (100, -100) for each point
                distances_from_100 = np.sqrt((100 - df_quadrant['prevalence']) ** 2 + (-100 - df_quadrant['overlap']) ** 2)

                angles = np.degrees(np.arctan2(df_quadrant['overlap'], df_quadrant['prevalence'])).abs()
                angles_from_horizontal = np.where(angles > 90, 180 - angles, angles)
                angles_from_horizontal[(df_quadrant['prevalence'] == 0) & (df_quadrant['overlap'] == 0)] = 0
                df_quadrant['updated_angle_degrees'] = np.where(angles_from_horizontal > 45, 45 - (angles_from_horizontal - 45),
                                                                angles_from_horizontal)

                # Add a new column for distances from origin
                df_quadrant['distance_from_origin'] = distances
                df_quadrant['distance_from_(100,-100)'] = distances_from_100

                df_quadrant['Interim_score_Bottom_Right_Quadrant'] = 45 * df_quadrant['distance_from_origin'] + df_quadrant[
                    'updated_angle_degrees']
                
                # Calculate the angles using the provided formula
                angles = np.arctan(
                    np.abs(-100 - df_quadrant['overlap']) / np.abs(100 - df_quadrant['prevalence'])) * 180 / np.pi

                # Create a new column for calculated angles
                df_quadrant['angle_from_(100,-100)'] = angles

                df_quadrant['updated_angle_degrees_(100,-100)'] = np.where(angles > 45, (90 - angles), angles)

                df_quadrant['Interim_score_any_other_Quadrant'] = -45 * df_quadrant['distance_from_(100,-100)'] + df_quadrant[
                    'updated_angle_degrees_(100,-100)']

                df_quadrant['interim_score'] = np.where(df_quadrant['quadrant'] == 'Bottom_right',
                                                        df_quadrant['Interim_score_Bottom_Right_Quadrant'],
                                                        df_quadrant['Interim_score_any_other_Quadrant'])
                
                df_quadrant['category'] = ''
                # Loop through the dictionary and assign keys to the rows
                for key, words in result_dict.items():
                    df_quadrant.loc[df_quadrant.index.isin(words), 'category'] = key
                
                sorted_df = df_quadrant.sort_values(by=['quadrant', 'interim_score'], ascending=[True, False])
                sorted_df['Overall Ranking Within Quadrant'] = sorted_df.groupby('quadrant').cumcount() + 1

                Top_right = sorted_df[sorted_df['quadrant'] == 'Top_right'][['category', 'Overall Ranking Within Quadrant']]
                Top_right = Top_right.sort_values(by=['category', 'Overall Ranking Within Quadrant'], ascending=[True, True])
                Top_right['Ranking Within Category'] = Top_right.groupby('category').cumcount() + 1

                Bottom_right = sorted_df[sorted_df['quadrant'] == 'Bottom_right'][['category', 'Overall Ranking Within Quadrant']]
                Bottom_right = Bottom_right.sort_values(by=['category', 'Overall Ranking Within Quadrant'], ascending=[True, True])
                Bottom_right['Ranking Within Category'] = Bottom_right.groupby('category').cumcount() + 1

                Top_left = sorted_df[sorted_df['quadrant'] == 'Top_left'][['category', 'Overall Ranking Within Quadrant']]
                Top_left = Top_left.sort_values(by=['category', 'Overall Ranking Within Quadrant'], ascending=[True, True])
                Top_left['Ranking Within Category'] = Top_left.groupby('category').cumcount() + 1

                Bottom_left = sorted_df[sorted_df['quadrant'] == 'Bottom_left'][['category', 'Overall Ranking Within Quadrant']]
                Bottom_left = Bottom_left.sort_values(by=['category', 'Overall Ranking Within Quadrant'], ascending=[True, True])
                Bottom_left['Ranking Within Category'] = Bottom_left.groupby('category').cumcount() + 1


        #---------------------------------------------------------------------------------------------------------------------------------
            with col2:
                selected_quadrant = st.selectbox("Select Quadrant:", tuple(sorted_df['quadrant'].unique()))
                match={'Top_right':Top_right,'Bottom_right':Bottom_right,'Top_left':Top_left,'Bottom_left':Bottom_left}

                if selected_quadrant:
                    selected_quadrant = match[selected_quadrant]
                    segments_list = selected_quadrant.index.tolist()
                
                Quadrant_selection = filter_condition(df_filtered,segments_list)

                for item in segments_list:
                    Quadrant_selection[item] = Quadrant_selection.apply(lambda row: 1 if any(
                        str(item) in val.split(',') if isinstance(val, str) else False for val in row) else 0, axis=1)

                overall_percentage = (len(Quadrant_selection) / len(df_filtered)) * 100
                st.write(f"Showing Data Percentage :",f"{overall_percentage:.2f}%")
                st.write(selected_quadrant)
            
    #------------------------------------------------------------------------------------------------------------------------
            selected_quadrant = selected_quadrant.reset_index()
            selected_quadrant = selected_quadrant.rename(columns={selected_quadrant.columns[0]: 'segments'})
            def twin_bar_chart(bar_df,index_df):
                fig, ax1 = plt.subplots(figsize=(12, 5))
                bars = ax1.bar(bar_df['Column Name'], bar_df['Percentage'], color="orange", alpha=0.7, label='df_demographics_Top3 (Bar Chart)')
                ax1.set_xlabel('Demographics',color='white')
                ax1.set_ylabel('Segment- Percentage', color='white')
                ax1.set_title('Profiling',color='white')
                plt.xticks(rotation=90,color='white')
                for bar, value in zip(bars, bar_df['Percentage']):
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom',
                            color='white')
        #
                # # # Create a twin right y-axis for df2
                ax2 = ax1.twinx()
                line = ax2.plot(bar_df['Column Name'], index_df['index'], marker='o', linestyle='-', color='red', label='index (Line Chart)')
                ax2.set_ylabel('index', color='red')
                plt.yticks(color='white')
                #
                # # Add data labels for the points in the line chart (df2)
                for x, y in zip(bar_df['Column Name'], index_df['index']):
                    label = f'{y:.2f}'  # Format the label as desired
                    ax2.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='red',rotation=90)


                # Combine legends from both axes
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()

                fig.patch.set_alpha(0.0)
                ax1.set_facecolor('none')
                boundaries=['top','right','bottom','left']
                for boundary in boundaries:
                    ax2.spines[boundary].set_visible(False)

                return plt,ax1,ax2
            selected_segment = st.radio("Segmentation Selection:", ('Top_3_Segments_in_each_category', 'Custom_Selection'))
            if selected_segment == 'Top_3_Segments_in_each_category':
                # Create an empty dictionary to store the top three segments in each category
                top_segments_by_category = {}

                # Iterate through each category
                for category in selected_quadrant['category'].unique():
                    # Get the top three segments for the current category
                    top_segments = selected_quadrant[selected_quadrant['category'] == category]['segments'][:3].tolist()

                    # Store the top segments in the dictionary
                    top_segments_by_category[category] = top_segments
                concatenated_list = []

                # Iterate through the dictionary values and extend the list
                for values_list in top_segments_by_category.values():
                    concatenated_list.extend(values_list)

                filtered_df_Top3=df_filtered
                filtered_df_Top3 =filter_condition(filtered_df_Top3,concatenated_list)
                filtered_df_Top3_count=len(filtered_df_Top3)
                backup_filtered_df_Top3=filtered_df_Top3
                filtered_df_Top3=filtered_df_Top3[['income','gender','age_category']]
                filtered_df_Top3=demographics(filtered_df_Top3)

                index=pd.merge(filtered_df_Top3,demographics_df,on='Column Name',how='inner')
                
                index['index'] = (index.iloc[:,1] / index.iloc[:,2]*100)
                
                twin_bar_chart(filtered_df_Top3,index)
                plt.tight_layout()

                st.pyplot(plt)
                for key, values in top_segments_by_category.items():
                    df = pd.DataFrame({key: values})
                    st.write(df)
                # st.write('data count: ',filtered_df_Top3_count)
                col11,col12,col13=st.columns((3))
                with col11:
                    age_category=st.multiselect('select age range ',backup_filtered_df_Top3['age_category'].unique())
                with col12:
                    gender=st.multiselect('select gender ',backup_filtered_df_Top3['gender'].unique())
                with col13:
                    income=st.multiselect('select income  ',backup_filtered_df_Top3['income'].unique())
                df_Selection=backup_filtered_df_Top3.query('age_category ==@age_category & gender==@gender & income==@income')

                # st.write(df_Selection)

                
                    



            else:
                selected_segments = st.multiselect("Select segments:", segments_list)
                if len(selected_segments)>0:
                    custom_df=df_filtered
                    custom_df =filter_condition(custom_df,selected_segments)
                    backup_custom_df=custom_df
                    custom_df=custom_df[['income','gender','age_category']]
                    # custom_df_count=len(custom_df)
                    custom_df=demographics(custom_df)

                    index=pd.merge(custom_df,demographics_df,on='Column Name',how='inner')
                    
                    index['index'] = (index.iloc[:,1] / index.iloc[:,2]*100)
                    twin_bar_chart(custom_df,index)
                    plt.tight_layout()
                    st.pyplot(plt)
                    col11,col12,col13=st.columns((3))
                    with col11:
                        age_category=st.multiselect('select age range ',backup_custom_df['age_category'].unique())
                    with col12:
                        gender=st.multiselect('select gender ',backup_custom_df['gender'].unique())
                    with col13:
                        income=st.multiselect('select income  ',backup_custom_df['income'].unique())
                    df_Selection=backup_custom_df.query('age_category ==@age_category & gender==@gender & income==@income')

                    # st.write(df_Selection)
                    # st.write('final filtered data count: ', len(df_Selection))
                    # st.write(backup_custom_df)
                else:
                    selected_segments=[]
            st.write('final filtered data count: ',len(df_Selection))
            with st.expander("View Data"):
                st.write(df_Selection)
            csv=df_Selection.to_csv(index=False).encode('utf-8')
            st.download_button("Download Data",data=csv,file_name="Result.csv")
                
            return scatter_plot(merged_op),plt,Quadrant_selection,segments_list,selected_columns
        # Plot_chart_method(df_filtered,selected_segments,result_dict)
#---------------------------------------------------------------------------------------------------------------------------------
        def network_graph_method(df_filtered,selected_columns):
            network_df=df_filtered[selected_columns]
            network_df=filter_condition(network_df,selected_segments)
            for col in network_df.columns:
                network_df[col] = network_df[col].apply(lambda x: ','.join([val for val in str(x).split(',') if val in selected_segments]))

            def concat_non_blank(row):
                return ','.join([val for val in row if val != ''])

        # Apply the custom function to each row
            network_df['concatenated'] = network_df.apply(concat_non_blank, axis=1)
            
            # def concatenate_without_nan(row):
            #     return ','.join([str(value) for value in row if not pd.isna(value)])
            #         # # Apply the custom function column-wise to create a new 'Concatenated' column
            # network_df['concatenated'] = network_df.apply(concatenate_without_nan, axis=1)
            network_df.drop(['interests', 'brands_visited', 'place_categories', 'geobehaviour'],axis=1,inplace=True)
            # def concat_matching_items(column, item_list):
            #     return ','.join([item for item in item_list if column and item in column])
            # network_df['concatenated'] = network_df['concatenated'].apply(lambda x: concat_matching_items(x, segments_list_req))
            def count_elements(s):
                return len(s.split(','))

            # Filter rows where the 'interests' column has more than one element
            network_df = network_df[network_df['concatenated'].apply(count_elements) > 1]
            network_df['concatenated_items'] = network_df['concatenated'].str.split(',')
            nodes_df = pd.DataFrame(network_df['concatenated_items'].explode().unique(), columns=['Node'])
            nodes_df = nodes_df[nodes_df['Node'] != '']

            edges = []

            for _, row in network_df.iterrows():
                nodes = row['concatenated_items']
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        edges.append((nodes[i], nodes[j]))

            edges_df = pd.DataFrame(edges, columns=['Source', 'Target'])
            network = Network(notebook=True, width='800px', height='600px')

    #         # Add nodes
            for _, node in nodes_df.iterrows():
                network.add_node(node['Node'])

            # Add edges
            for _, edge in edges_df.iterrows():
                network.add_edge(edge['Source'], edge['Target'])

            edges_df = pd.DataFrame(np.sort(edges_df.values, axis=1), columns=edges_df.columns)

            edges_df['Value']=1
            edges_df=edges_df.groupby(['Source','Target'],sort=False,as_index=False).sum()
            edges_df = edges_df.sort_values(by='Value', ascending=False)

            G = nx.from_pandas_edgelist(edges_df, source='Source', target='Target', edge_attr='Value', create_using=nx.Graph())
            net = Network(notebook=True, width='1000px', height='700px', bgcolor='#22222E', font_color='white')
            node_degree = dict(G.degree)
            communities = community_louvain.best_partition(G)
            nx.set_node_attributes(G, node_degree, 'size')
            nx.set_node_attributes(G, communities, 'group')

            net.from_nx(G)
            # net.repulsion(node_distance=9000, spring_length=20000)
            net.set_options("""
                var options = {
                "physics": {
                    "enabled": false
                }
                }
            """)
            with col1:
                net.show('com_net_network_graph.html')
                st.components.v1.html(open("com_net_network_graph.html").read(), height=700, width=1000)
                
            communities_df = pd.DataFrame.from_dict(communities, orient='index', columns=['Community'])
            communities_df.reset_index(inplace=True)
            communities_df.rename(columns={'index': 'Segments'}, inplace=True)
            communities_df.sort_values(by='Community', ascending=True, inplace=True)
            communities_df.reset_index(drop=True, inplace=True)

            #-----------------------------------------------------------------------------------------------------------------------------

            degree_dict=nx.degree_centrality(G)
            degree_df=pd.DataFrame.from_dict(degree_dict,orient="index",columns=['centrality']).sort_values(by='centrality', ascending=False)
            col4,col5,col6=st.columns((3))
            
            def bar_chart_centrality(df,y_label,title,bar_color):
                fig, ax = plt.subplots(figsize=(15, 9))  # Adjust the figure size as needed

                # Use barplot from Matplotlib to create the bar chart
                plt.barh(df.index, df['centrality'], color=bar_color)

                # Add labels and title
                plt.xlabel('Segments',color='white')
                plt.ylabel(y_label,color='white')
                # plt.title(title,color='white')
                st.markdown(title)

                # Rotate the x-axis labels for better readability
                plt.xticks(rotation=90,color='white')
                plt.yticks(color='white')


                # Show the plot
                plt.tight_layout()  # Ensures labels fit within the figure
                plt.show()
                fig.patch.set_alpha(0.0)
                ax.set_facecolor('none')
                boundaries=['top','right','bottom','left']
                for boundary in boundaries:
                    ax.spines[boundary].set_visible(False)
                return plt
            with col4:
                degree_df=bar_chart_centrality(degree_df,'Degree Centrality Value','Degree Centrality of Segments (represent how many charactors)','orange')

                plt.tight_layout()  
                st.pyplot(plt)

            with col5:
                betweenness_dict=nx.betweenness_centrality(G)
                betweenness_df=pd.DataFrame.from_dict(betweenness_dict,orient="index",columns=['centrality']).sort_values(by='centrality', ascending=False)
                betweenness_df=bar_chart_centrality(betweenness_df,'Betweenness Centrality Value','Betweenness Centrality of Segments (Represent how connects communities)','green')

                plt.tight_layout()  
                st.pyplot(plt)

            with col6:
                clossness_dict=nx.closeness_centrality(G)
                betweenness_df=pd.DataFrame.from_dict(clossness_dict,orient="index",columns=['centrality']).sort_values(by='centrality', ascending=False)
                betweenness_df=bar_chart_centrality(betweenness_df,'Clossness Centrality Value','Clossness Centrality of Segments (Represent segments that are the closest)','purple')

                plt.tight_layout()  
                st.pyplot(plt)

            return plt,communities_df
        # network_graph_method(df_filtered,selected_columns)
        if approach == 'Plot chart only':
            # required_list = [value for values in result_dict.values() for value in values]
            Plot_chart_method(df_filtered,selected_segments,result_dict)

        elif approach == 'Relationship Graph only':
            net=network_graph_method(df_filtered,selected_columns)
            non2,communities_df=net
            with col2:
                st.write(communities_df)
                seg_selection=st.multiselect('Chooose Segments: ',communities_df.Segments.tolist())
                seg_selection_df=filter_condition(df_filtered,seg_selection)
            st.write('final filtered data count: ',len(seg_selection_df))
            with st.expander("View Data"):
                st.write(seg_selection_df)
            csv=seg_selection_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Data",data=csv,file_name="Result.csv")

            # with col1:
            #     non2

        elif approach == 'Plot chart first and Relationship Graph after':
            plot_chart=Plot_chart_method(df_filtered,selected_segments,result_dict)
            non1,non2,Quadrant_selection,segments_list,selected_columns=plot_chart

            def filter_items(cell):
                if pd.isna(cell):
                    return None
                items = cell.split(',')
                filtered_items = [item for item in items if item in segments_list]
                if filtered_items:
                    return ','.join(filtered_items)
                return None

            for col in selected_columns:
                Quadrant_selection[col] = Quadrant_selection[col].apply(filter_items)
            network_graph_method(Quadrant_selection,selected_columns)

        elif approach == 'Relationship Graph first and Plot Chart after':
            net=network_graph_method(df_filtered,selected_columns)
            non2,communities_df=net
            with col2:
                st.write(communities_df)
                community_selection_option=st.selectbox('Select option: ',('Community wise analyze','custom analyze'))
            if community_selection_option=='Community wise analyze':
                with col2:
                    community_selection=st.radio('Select community: ',communities_df.Community.unique())
                communities_df=communities_df[communities_df['Community']==community_selection]
                communities_df_list = communities_df.Segments.tolist()
                communities_df=filter_condition(df_filtered,communities_df_list)

                Plot_chart_method(communities_df,communities_df_list,result_dict)
            else:
                # custom_analyze_list = communities_df.Segments.tolist()
                all_values = [value for values in result_dict.values() for value in values]
                # result = [item.split(":")[1].strip(' "') for item in communities_df_list1]
                custom_analyze_list=st.multiselect("Select segments: ", all_values)
                # if len(custom_analyze_list)>0:
                custom_analyze_df=filter_condition(df_filtered,custom_analyze_list)
                Plot_chart_method(custom_analyze_df,custom_analyze_list,result_dict)
if __name__=='__main__':

    main()