import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import spacy
from spacy import displacy
import networkx as nx
import community as community_louvain

import re
# from adjustText import adjust_text
# from PIL import Image
# import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
# from bs4 import BeautifulSoup
import io
st.set_page_config(page_title='AFFIX PRODUCT EXPLORER',page_icon=':bar_chart:',layout='wide')

# st.set_page_config(page_title='AFFIX PRODUCT EXPLORER',page_icon=':bar_chart:',layout='wide')
custom_css = """
<style>
body {
    background-color: #22222E; 
    secondary-background {
    background-color: #FA55AD; 
    padding: 10px; 
}
</style>
"""
st.write(custom_css, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)

# Use the secondary background color in your app
# st.markdown('<div class="secondary-background">This is a section with a custom secondary background color.</div>', unsafe_allow_html=True)

def twin_bar_chart(bar_df,index_df):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    bars = ax1.bar(bar_df['Column_Name'], bar_df['Percentage'], color="orange", alpha=0.7, label='df_demographics_Top3 (Bar Chart)')
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
    line = ax2.plot(bar_df['Column_Name'], index_df['index'], marker='o', linestyle='-', color='red', label='index (Line Chart)')
    ax2.set_ylabel('index', color='red')
    plt.yticks(color='white')
    #
    # # Add data labels for the points in the line chart (df2)
    for x, y in zip(bar_df['Column_Name'], index_df['index']):
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




def main():
    st.title(" :bar_chart: AFFIX PRODUCT EXPLORER")
    st.sidebar.image('AFFIXCON-LOGO.png')
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
            select_all_segments = st.checkbox("Select All Segments",value=True)
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
            select_all_segments = st.checkbox("Select All Segments",value=True)

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

        df_backup=df.copy()
        for item in selected_segments:
            df_backup[item] = df_backup.apply(lambda row: 1 if any(
                str(item) in val.split(',') if isinstance(val, str) else False for val in row) else 0, axis=1)
        df_backup.drop(selected_columns,axis=1,inplace=True)
        # df_backup=df_backup.drop(columns=['income','age','gender'])
        zero_columns = df_backup.columns[(df_backup == 0).all()]
        df_backup = df_backup.drop(columns=zero_columns)

        columns_for_custom_columns = list(filter(lambda value: value in selected_segments, df_backup.columns))
        df_backup=df_backup[columns_for_custom_columns]


        with col3:
            st.write("Your Matched data count is: ",len(df))
        bin_edges = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,  float('inf')]  
        bin_labels = ['Age less than 20', 'Age 20-24', 'Age 25-29','Age 30-34', 'Age 35-39','Age 40-44', 'Age 45-49','Age 50-54', 'Age 55-59','Age 60-64', 'Age 65-69','Age 70-74', 'Age 75-79','Age 80-84','Age 85+']  
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
            df = pd.DataFrame({'Column_Name': df.columns, 'Percentage': percentages})
            return df
        
        demographics_df=demographics(demographics_df)
        demographics_df2=demographics_df.copy()
        # st.write(demographics_df)
        custom_order = [
            "Very Low (Under $20,799)","Low ($20,800 - $41,599)","Below Average ($41,600 - $64,999)","Average ($65,000 - $77,999)",
            "Above Average ($78,000 - $103,999)","High ($104,000 - $155,999)","Very high ($156,000+)","unknown_income",
            "Male","Female","unknown_gender",
            'Age less than 20', 'Age 20-24', 'Age 25-29','Age 30-34', 'Age 35-39','Age 40-44', 'Age 45-49',
            'Age 50-54', 'Age 55-59','Age 60-64', 'Age 65-69','Age 70-74', 'Age 75-79','Age 80-84','Age 85+'
        ]

                # Create a custom sort key function
        def custom_sort_key(item):
            try:
                return custom_order.index(item)
            except ValueError:
                return len(custom_order)

        # Apply the custom sort key function to the "Column_Name" column
        demographics_df2['Custom_Sort_Key'] = demographics_df2['Column_Name'].map(custom_sort_key)
        demographics_df2 = demographics_df2.sort_values('Custom_Sort_Key')
        demographics_df2 = demographics_df2.reset_index(drop=True)
        demographics_df2 = demographics_df2.drop(columns=['Custom_Sort_Key'])

        demographics_df1=demographics_df.copy()

        def update_dem(Column_Name):
            if 'Age' in Column_Name:
                return 'age_' + Column_Name
            elif Column_Name in ['Male', 'Female']:
                return 'gender_' + Column_Name
            elif '$' in Column_Name:
                return 'income_' + Column_Name
            else:
                return Column_Name

        def top_dem_selection(dem_df):
            dem_df['Column_Name'] = dem_df['Column_Name'].apply(update_dem)
            dem_df[['category', 'value', 'unknown']] = dem_df['Column_Name'].str.extract(r'([a-zA-Z ]+)([\$\d]+)?(unknown)?')
            dem_df = dem_df[~dem_df['category'].isin(['unknown'])]
            # Group the DataFrame by 'category' and find the row with the highest percentage
            highest_rows = dem_df.groupby('category')['Percentage'].idxmax()

            # Filter the DataFrame to get the rows with the highest percentages
            top_demographics = dem_df.loc[highest_rows]

            # Drop the unnecessary columns
            top_demographics = top_demographics.drop(columns=['value', 'unknown',"category",'Column_Name'])
            return top_demographics

        with col3:
            with st.expander("View Top Demographics"):
                st.write(top_dem_selection(demographics_df1))
        
        def bar_chart(df,x,y,x_label,y_label,title):
            fig, ax = plt.subplots(figsize=(12, 5))
            bars = plt.bar(df[x], df[y], color='#0083B8')
            plt.xlabel(x_label,color='white')
            plt.ylabel(y_label,color='white')
            plt.title(title,color='white')
            plt.xticks(rotation=90,color='white')
            plt.yticks(color='white')


            for bar, value in zip(bars, df[y]):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', rotation=90,color='white')

            fig.patch.set_alpha(0.0)
            ax.set_facecolor('none')
            boundaries=['top','right','bottom','left']
            for boundary in boundaries:
                ax.spines[boundary].set_visible(False)
            return plt
        with col3:
            with st.expander("View All Demographics"):
                bar_plot = bar_chart(demographics_df2,"Column_Name","Percentage","Demographics","Percentage (%)","Percentage of Each Demographics")
                st.pyplot(bar_plot)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

            # Provide a download button for the chart image
                st.download_button("Download Demographics", data=buffer, file_name="Demographics.png")
        with col3:
            select_all_segments=st.selectbox('Selection Demographics',('Select All Demographics','Custom Demographics'))
            
            if select_all_segments=='Select All Demographics':
                df_filtered=df
        
            else:
                with col3:
                    selected_age_category = st.multiselect("Select Age Category", ('less than 20', 'Age 20-24', 'Age 25-29','Age 30-34', 'Age 35-39','Age 40-44', 'Age 45-49','Age 50-54', 'Age 55-59','Age 60-64', 'Age 65-69','Age 70-74', 'Age 75-79','Age 80-84','85+'))
                    selected_gender_category = st.multiselect("Select Gender Category", ('Male','Female'))
                    selected_income_category = st.multiselect("Select Income Category", ("Very Low (Under $20,799)", "Low ($20,800 - $41,599)", "Below Average ($41,600 - $64,999)","Average ($65,000 - $77,999)", "Above Average ($78,000 - $103,999)","High ($104,000 - $155,999)","Very high ($156,000+)"))

                    filter_criteria=selected_age_category+selected_gender_category+selected_income_category
                    df_filtered = df[df.isin(filter_criteria).any(axis=1)]

            with col3:
                st.write("Demographics matched data count is: ",len(df_filtered))
        # st.write(df_filtered)
#-----------------------------------------------------------------------------------------------------------------------------------

        # @st.cache_data 
        def Plot_chart_method(df_filtered,selected_segments,result_dict,skip_calculation=True):
            selected_columns = ['interests', 'brands_visited', 'place_categories', 'geobehaviour']
            segments_filtered_df=df_filtered[selected_columns]

            for item in selected_segments:
                segments_filtered_df[item] = segments_filtered_df.apply(lambda row: 1 if any(
                    str(item) in val.split(',') if isinstance(val, str) else False for val in row) else 0, axis=1)
            segments_filtered_df.drop(selected_columns,axis=1,inplace=True)
            zero_columns = segments_filtered_df.columns[(segments_filtered_df == 0).all()]
            segments_filtered_df = segments_filtered_df.drop(columns=zero_columns)
            

            if skip_calculation:
                prevalence = segments_filtered_df.agg(lambda col: round(col.mean() * 100, 2)).reset_index()
                prevalence.columns = ['Segments','prevalence']
                segments_filtered_df = segments_filtered_df.replace(0, 'no data')

                csv=prevalence.to_csv(index=False).encode('utf-8')
                # st.download_button("prevalence",data=csv, file_name="prevalence.csv")
                overlap=segments_filtered_df
                sum_non_zero = overlap.apply(lambda col: col[col == 1].sum(),axis=1)-1
                count=len(overlap.columns)
                row_mean = (sum_non_zero/count)*100
                for column in overlap.columns:
                    overlap[column] = overlap[column].mask(overlap[column] == 1, row_mean, axis=0) 

                sum_non_data_column = overlap.apply(lambda col: col[col != 'no data' ].sum(),axis=0)
                count_non_data_column=overlap.apply(lambda col: col[col != 'no data' ].count(),axis=0)
                column_mean=round(sum_non_data_column/count_non_data_column,2)
                overlap=pd.DataFrame([column_mean],columns=overlap.columns).fillna(0).transpose().reset_index()
                overlap.columns=['Segments','overlap']
                # st.write('overlap',overlap)
                csv=overlap.to_csv(index=False).encode('utf-8')
                # st.download_button("overlap",data=csv, file_name="overlap.csv")
                merged_op = pd.merge(prevalence, overlap,on='Segments',how='inner').set_index("Segments").transpose()
                merged_op['_avg'] = merged_op[merged_op.columns].mean(axis=1)
                merged_op['_std'] = merged_op[merged_op.columns].std(axis=1)
                # st.write('merged_op',merged_op)

                new_row_values = []
                for _, row in merged_op.iterrows():
                    # st.write(row)
                    new_row = []
                    for column in merged_op.columns[:-2]:
                        # st.write(float((row[column] - row['_avg']) / row['_std']))
                        new_row.append(float((row[column] - row['_avg']) / row['_std']))
                        new_row_values.append(new_row)
                df_new_row = pd.DataFrame(new_row_values, columns=merged_op.columns[:-2]).drop_duplicates()

                merged_op=df_new_row.transpose()
                merged_op.columns=['prevalence','overlap']
                merged_op=merged_op.transpose()

            else:
                result_dict = {key: [value for value in result_dict[key] if value in segments_filtered_df.columns] for key in result_dict}
                result_dict = {key: values for key, values in result_dict.items() if values}

                new_order = ["BRAND VISITED", "PLACE CATEGORIES", "INTERESTS", "GEO BEHAVIOUR"]
                items = list(result_dict.keys())
                sorted_items = sorted(items, key=lambda x: new_order.index(x))

                new_columns_order=[]
                for item in sorted_items:
                    new_columns_order.extend(result_dict[item])
                segments_filtered_df = segments_filtered_df[new_columns_order]
            # st.write(segments_filtered_df)

                prevalence = segments_filtered_df.agg(lambda col: round(col.mean() * 100, 2)).reset_index()
            # st.write(prevalence)

                prevalence.columns = ['Segments','prevalence']
                segments_filtered_df = segments_filtered_df.replace(0, 'no data')
            # st.write(segments_filtered_df)


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
            # st.write(merged_op)
                result_dict = {key: result_dict[key] for key in new_order if key in result_dict}
                for key, value in result_dict.items():
                    if key not in new_order:
                        result_dict[key] = value
                new_row_values = []
            # st.write(result_dict)

                for _, row in merged_op.iterrows():
                    new_row = []
                    for key, original_cols in result_dict.items():
                        new_row.extend((row[original_cols] - row[key + '_avg']) / row[key + '_std'])
                    new_row_values.append(new_row)
                df_new_row = pd.DataFrame(new_row_values, columns=merged_op.columns[:-len(result_dict) * 2])

                merged_op = df_new_row
                merged_op = merged_op.fillna(0)
                merged_op.index = merged_op.index.map({0: 'prevalence', 1: 'overlap'})
        
        # st.write(merged_op) 
    # merged_op=Plot_chart_method(df_filtered,selected_segments,result_dict,True)
    # st.write('all cat',merged_op.transpose())

    # merged_op=Plot_chart_method(df_filtered,selected_segments,result_dict,False)
    # st.write('withing cat',merged_op.transpose())        
            df_quadrant = merged_op.transpose()
            df_quadrant['quadrant'] = 'No Quadrant'
            df_quadrant.loc[(df_quadrant['prevalence'] > 0) & (df_quadrant['overlap'] > 0), 'quadrant'] = 'Top_right'
            df_quadrant.loc[(df_quadrant['prevalence'] < 0) & (df_quadrant['overlap'] > 0), 'quadrant'] = 'Top_left'
            df_quadrant.loc[(df_quadrant['prevalence'] < 0) & (df_quadrant['overlap'] < 0), 'quadrant'] = 'Bottom_left'
            df_quadrant.loc[(df_quadrant['prevalence'] > 0) & (df_quadrant['overlap'] < 0), 'quadrant'] = 'Bottom_right'
            
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


            return prevalence,Top_left,Top_right,Bottom_left,Bottom_right,merged_op

        def oppottunities(quadrant):
            selected_quadrant = match[quadrant]
            segments_list = selected_quadrant.index.tolist()
            Quadrant_selection = filter_condition(df_filtered,segments_list)
            top_demog=Quadrant_selection[['income','gender','age_category']]
            top_demog=demographics(top_demog)
            top_demog=top_dem_selection(top_demog)
            st.write('Top Demographics',top_demog)
            for item in segments_list:
                Quadrant_selection[item] = Quadrant_selection.apply(lambda row: 1 if any(
                    str(item) in val.split(',') if isinstance(val, str) else False for val in row) else 0, axis=1)
            overall_percentage = (len(Quadrant_selection) / len(df_filtered)) * 100
            st.write(f"Showing Data Percentage :",f"{overall_percentage:.2f}%")
            st.write(selected_quadrant)
            csv=selected_quadrant.to_csv(index=True).encode('utf-8')
            # st.download_button("Download oppotiunity",data=csv, file_name="oppotiunity.csv")
            
    #------------------------------------------------------------------------------------------------------------------------
            selected_quadrant = selected_quadrant.reset_index()
            selected_quadrant = selected_quadrant.rename(columns={selected_quadrant.columns[0]: 'segments'})
            # st.write(Top_right)
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

                filtered_df_Top3_backup=filtered_df_Top3.copy()
                from collections import Counter
                vocab=Counter()
                filtered_df_Top3_backup['Concatenated'] = filtered_df_Top3_backup[['interests', 'brands_visited', 'place_categories',"geobehaviour"]].apply(lambda row: ','.join(str(val) for val in row), axis=1)
                for line in filtered_df_Top3_backup['Concatenated']:
                    vocab.update(line.split(","))
                vocab = {key: vocab[key] for key in columns_for_custom_columns}
                vocab = pd.DataFrame(list(vocab.items()), columns=["Segment", "Count"]).sort_values("Count",ascending=False)
                fig, ax = plt.subplots(figsize=(12, 5))
                bars = plt.barh(vocab["Segment"], vocab["Count"], color='#0083B8')
                plt.xlabel("Counts",color='white')
                plt.ylabel("Segment",color='white')
                plt.title("Segments Counts",color='white')
                plt.xticks(color='white')
                plt.yticks(color='white')

                fig.patch.set_alpha(0.0)
                ax.set_facecolor('none')
                st.pyplot(plt)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                st.download_button("Download segment count", data=buffer, file_name="segment count.png")

                filtered_df_Top3_count=len(filtered_df_Top3)
                backup_filtered_df_Top3=filtered_df_Top3
                filtered_df_Top3=filtered_df_Top3[['income','gender','age_category']]
                filtered_df_Top3=demographics(filtered_df_Top3)

                index=pd.merge(filtered_df_Top3,demographics_df,on='Column_Name',how='inner')
                
                index['index'] = (index.iloc[:,1] / index.iloc[:,2]*100)
                
                twin_bar_chart(filtered_df_Top3,index)
                plt.tight_layout()

                st.pyplot(plt)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                # Provide a download button for the chart image
                # st.download_button("Download Index Demographics", data=buffer, file_name="Demographics Index.png")                    
                st.markdown('Top 3 Segments')
                for key, values in top_segments_by_category.items():
                    df = pd.DataFrame({key: values})
                    st.write(df)
                # col11,col12,col13=st.columns((3))
                # with col11:
                age_category=st.multiselect('select age range ',backup_filtered_df_Top3['age_category'].unique())
                # with col12:
                gender=st.multiselect('select gender ',backup_filtered_df_Top3['gender'].unique())
                # with col13:
                income=st.multiselect('select income  ',backup_filtered_df_Top3['income'].unique())
                df_Selection=backup_filtered_df_Top3.query('age_category ==@age_category & gender==@gender & income==@income')

                # csv=df_Selection.to_csv(index=True).encode('utf-8')
                # # st.download_button("Download final filtered table",data=csv, file_name="Rapid information dissemination segments.csv")

            else:
                selected_segments = st.multiselect("Select segments:", segments_list)
                if len(selected_segments)>0:
                    custom_df=df_filtered
                    custom_df =filter_condition(custom_df,selected_segments)

                    custom_df_backup=custom_df.copy()
                    from collections import Counter
                    vocab=Counter()
                    custom_df_backup['Concatenated'] = custom_df_backup[['interests', 'brands_visited', 'place_categories',"geobehaviour"]].apply(lambda row: ','.join(str(val) for val in row), axis=1)
                    for line in custom_df_backup['Concatenated']:
                        vocab.update(line.split(","))
                    vocab = {key: vocab[key] for key in columns_for_custom_columns}
                    vocab = pd.DataFrame(list(vocab.items()), columns=["Segment", "Count"]).sort_values("Count",ascending=False)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    bars = plt.barh(vocab["Segment"], vocab["Count"], color='#0083B8')
                    plt.xlabel("Counts",color='white')
                    plt.ylabel("Segment",color='white')
                    plt.title("Segments Counts",color='white')
                    plt.xticks(color='white')
                    plt.yticks(color='white')

                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor('none')
                    st.pyplot(plt)
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    st.download_button("Download segment count1", data=buffer, file_name="segment count.png")

                    backup_custom_df=custom_df
                    custom_df=custom_df[['income','gender','age_category']]
                    # custom_df_count=len(custom_df)
                    custom_df=demographics(custom_df)

                    index=pd.merge(custom_df,demographics_df,on='Column_Name',how='inner')
                    
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
                    # csv=df_Selection.to_csv(index=True).encode('utf-8')
                    # st.download_button("Download final filtered table3",data=csv, file_name="Rapid information dissemination segments.csv")

                else:
                    selected_segments=[]
            st.write('final filtered data count: ',len(df_Selection))
            st.write(df_Selection)
            csv=df_Selection.to_csv(index=False).encode('utf-8')
            st.download_button("Download final filtered table",data=csv, file_name="final filtered data.csv")
            if selected_segment == 'Top_3_Segments_in_each_category':
                return top_demog,overall_percentage,selected_quadrant,df,df_Selection
            else:
                return top_demog,overall_percentage,selected_quadrant,df_Selection
            

        def oppottunities1(quadrant):
            selected_quadrant = match[quadrant]
            segments_list = selected_quadrant.index.tolist()
            Quadrant_selection = filter_condition(df_filtered,segments_list)
            top_demog=Quadrant_selection[['income','gender','age_category']]
            top_demog=demographics(top_demog)
            top_demog=top_dem_selection(top_demog)
            st.write('Top Demographics',top_demog)
            for item in segments_list:
                Quadrant_selection[item] = Quadrant_selection.apply(lambda row: 1 if any(
                    str(item) in val.split(',') if isinstance(val, str) else False for val in row) else 0, axis=1)
            overall_percentage = (len(Quadrant_selection) / len(df_filtered)) * 100
            st.write(f"Showing Data Percentage1 :",f"{overall_percentage:.2f}%")
            st.write(selected_quadrant)
            csv=selected_quadrant.to_csv(index=True).encode('utf-8')
            # st.download_button("Download oppotiunity",data=csv, file_name="oppotiunity.csv")
            
    #------------------------------------------------------------------------------------------------------------------------
            selected_quadrant = selected_quadrant.reset_index()
            selected_quadrant = selected_quadrant.rename(columns={selected_quadrant.columns[0]: 'segments'})
            # st.write(Top_right)
            selected_segment = st.radio("Segmentation Selection1:", ('Top_3_Segments_in_each_category1', 'Custom_Selection1'))
            if selected_segment == 'Top_3_Segments_in_each_category1':
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

                filtered_df_Top3_backup=filtered_df_Top3.copy()
                from collections import Counter
                vocab=Counter()
                filtered_df_Top3_backup['Concatenated'] = filtered_df_Top3_backup[['interests', 'brands_visited', 'place_categories',"geobehaviour"]].apply(lambda row: ','.join(str(val) for val in row), axis=1)
                for line in filtered_df_Top3_backup['Concatenated']:
                    vocab.update(line.split(","))
                vocab = {key: vocab[key] for key in columns_for_custom_columns}
                vocab = pd.DataFrame(list(vocab.items()), columns=["Segment", "Count"]).sort_values("Count",ascending=False)
                fig, ax = plt.subplots(figsize=(12, 5))
                bars = plt.barh(vocab["Segment"], vocab["Count"], color='#0083B8')
                plt.xlabel("Counts",color='white')
                plt.ylabel("Segment",color='white')
                plt.title("Segments Counts",color='white')
                plt.xticks(color='white')
                plt.yticks(color='white')

                fig.patch.set_alpha(0.0)
                ax.set_facecolor('none')
                st.pyplot(plt)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                st.download_button("Download segment count1", data=buffer, file_name="segment count.png")

                filtered_df_Top3_count=len(filtered_df_Top3)
                backup_filtered_df_Top3=filtered_df_Top3
                filtered_df_Top3=filtered_df_Top3[['income','gender','age_category']]
                filtered_df_Top3=demographics(filtered_df_Top3)

                index=pd.merge(filtered_df_Top3,demographics_df,on='Column_Name',how='inner')
                
                index['index'] = (index.iloc[:,1] / index.iloc[:,2]*100)
                
                twin_bar_chart(filtered_df_Top3,index)
                plt.tight_layout()

                st.pyplot(plt)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                # Provide a download button for the chart image
                # st.download_button("Download Index Demographics", data=buffer, file_name="Demographics Index.png")                    
                st.markdown('Top 3 Segments')
                for key, values in top_segments_by_category.items():
                    df = pd.DataFrame({key: values})
                    st.write(df)
                col11,col12,col13=st.columns((3))
                with col11:
                    age_category=st.multiselect('select age range1 ',backup_filtered_df_Top3['age_category'].unique())
                with col12:
                    gender=st.multiselect('select gender1 ',backup_filtered_df_Top3['gender'].unique())
                with col13:
                    income=st.multiselect('select income1  ',backup_filtered_df_Top3['income'].unique())
                df_Selection=backup_filtered_df_Top3.query('age_category ==@age_category & gender==@gender & income==@income')
                # csv=df_Selection.to_csv(index=True).encode('utf-8')
                # st.download_button("Download final filtered table2",data=csv, file_name="Rapid information dissemination segments.csv")

            else:
                selected_segments = st.multiselect("Select segments1:", segments_list)
                if len(selected_segments)>0:
                    custom_df=df_filtered
                    custom_df =filter_condition(custom_df,selected_segments)

                    custom_df_backup=custom_df.copy()
                    from collections import Counter
                    vocab=Counter()
                    custom_df_backup['Concatenated'] = custom_df_backup[['interests', 'brands_visited', 'place_categories',"geobehaviour"]].apply(lambda row: ','.join(str(val) for val in row), axis=1)
                    for line in custom_df_backup['Concatenated']:
                        vocab.update(line.split(","))
                    vocab = {key: vocab[key] for key in columns_for_custom_columns}
                    vocab = pd.DataFrame(list(vocab.items()), columns=["Segment", "Count"]).sort_values("Count",ascending=False)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    bars = plt.barh(vocab["Segment"], vocab["Count"], color='#0083B8')
                    plt.xlabel("Counts",color='white')
                    plt.ylabel("Segment",color='white')
                    plt.title("Segments Counts",color='white')
                    plt.xticks(color='white')
                    plt.yticks(color='white')

                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor('none')
                    st.pyplot(plt)
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    st.download_button("Download segment count1", data=buffer, file_name="segment count.png")

                    backup_custom_df=custom_df
                    custom_df=custom_df[['income','gender','age_category']]
                    # custom_df_count=len(custom_df)
                    custom_df=demographics(custom_df)

                    index=pd.merge(custom_df,demographics_df,on='Column_Name',how='inner')
                    
                    index['index'] = (index.iloc[:,1] / index.iloc[:,2]*100)
                    twin_bar_chart(custom_df,index)
                    plt.tight_layout()
                    st.pyplot(plt)
                    col11,col12,col13=st.columns((3))
                    with col11:
                        age_category=st.multiselect('select age range1 ',backup_custom_df['age_category'].unique())
                    with col12:
                        gender=st.multiselect('select gender1 ',backup_custom_df['gender'].unique())
                    with col13:
                        income=st.multiselect('select income1  ',backup_custom_df['income'].unique())
                    df_Selection=backup_custom_df.query('age_category ==@age_category & gender==@gender & income==@income')
                    # csv=df_Selection.to_csv(index=True).encode('utf-8')
                    # st.download_button("Download final filtered table1",data=csv, file_name="Rapid information dissemination segments.csv")

                else:
                    selected_segments=[]
            st.write('final filtered data count: ',len(df_Selection))
            st.write(df_Selection)
            csv=df_Selection.to_csv(index=False).encode('utf-8')
            st.download_button("Download final filtered table1",data=csv, file_name="final filtered data.csv")
            if selected_segment == 'Top_3_Segments_in_each_category1':
                return top_demog,overall_percentage,selected_quadrant,df,df_Selection
            else:
                return top_demog,overall_percentage,selected_quadrant,df_Selection
            
        
        
            


        with col1:
            oppotunity_options=st.radio("Oppotunity Selection:", ('More popular & More Competition', 
                                                                    'More popular & Less Competition',
                                                                    "Less popular & More Competition",
                                                                    'Less popular & Less Competition'))


        with col2:
            x = [1, -30, 3, 4, 15]
            y = [10, 12, -10, 6, 10]
            fig, ax = plt.subplots()
            ax.scatter(x, y, marker='o', color='b', alpha=0)
            ax.axhline(0, color='white', linewidth=2)  # Horizontal line at y=0
            ax.axvline(0, color='white', linewidth=2)  # Vertical line at x=0

            ax.text(2, 5, 'More popular & More Competition', fontsize=12, color='yellow')
            ax.text(-30, 5, 'Less popular & More Competition', fontsize=12, color='skyblue')
            ax.text(-30, -5, 'Less popular & Less Competition', fontsize=12, color='skyblue')
            ax.text(2, -5, 'More popular & Less Competition', fontsize=12, color='yellow')
            boundaries = ['top', 'right', 'bottom', 'left']
            ax.set_facecolor('none')
            for boundary in boundaries:
                ax.spines[boundary].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.patch.set_alpha(0)
            st.pyplot(plt)
        with col2:
            # with st.expander("With All Concat categories"):
            with st.expander("Custom Segments Selection"):
                Custom_Segments_Selections=st.multiselect('Select Segments: ',columns_for_custom_columns)
                if len(Custom_Segments_Selections)>0:
                    Custom_Segments_Selections_df=filter_condition(df_filtered,Custom_Segments_Selections)
                    Custom_Segments_Selections_df_backup=Custom_Segments_Selections_df.copy()
                    Custom_Segments_demog=Custom_Segments_Selections_df[['income','gender','age_category']]
                    Custom_Segments_demog=demographics(Custom_Segments_demog)
                    Custom_Segments_top_demog=Custom_Segments_demog.copy()
                    Custom_Segments_top_demog=top_dem_selection(Custom_Segments_top_demog)
                    st.write('Top Demographics',Custom_Segments_top_demog)
                    for item in Custom_Segments_Selections:
                        Custom_Segments_Selections_df[item] = Custom_Segments_Selections_df.apply(lambda row: 1 if any(
                            str(item) in val.split(',') if isinstance(val, str) else False for val in row) else 0, axis=1)
                    overall_percentage = (len(Custom_Segments_Selections_df) / len(df_filtered)) * 100
                    st.write(f"Showing Data Percentage :",f"{overall_percentage:.2f}%")

                    index=pd.merge(Custom_Segments_demog,demographics_df,on='Column_Name',how='inner')
                    index['index'] = (index.iloc[:,1] / index.iloc[:,2]*100)
                    twin_bar_chart(Custom_Segments_demog,index)
                    plt.tight_layout()
                    st.pyplot(plt)
                    col11,col12,col13=st.columns((3))
                    with col11:
                        age_category=st.multiselect('select_age_range ',Custom_Segments_Selections_df_backup['age_category'].unique())
                    with col12:
                        gender=st.multiselect('select_gender ',Custom_Segments_Selections_df_backup['gender'].unique())
                    with col13:
                        income=st.multiselect('select_income  ',Custom_Segments_Selections_df_backup['income'].unique())
                    custom_df_Selection=Custom_Segments_Selections_df_backup.query('age_category ==@age_category & gender==@gender & income==@income')
                    
                    st.write('final filtered data count: ',len(custom_df_Selection))
                    st.write(custom_df_Selection)
                    csv=custom_df_Selection.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Data",data=csv,file_name="Custom Result.csv")
                else:
                    pass




        prevalence,Top_left,Top_right,Bottom_left,Bottom_right,merged_op=Plot_chart_method(df_filtered,selected_segments,result_dict,True)

        match={'More popular & More Competition':Top_right,'More popular & Less Competition':Bottom_right,
                'Less popular & More Competition':Top_left,'Less popular & Less Competition':Bottom_left}
        prevalence = prevalence.sort_values(by='prevalence', ascending=False)

        with col3:
            with st.expander("View All Segments Distribution"):
                bar_plot_all_Segments = bar_chart(prevalence,"Segments","prevalence","Segments","Percentage of total(%)","Overall Segments Distribution")
                st.pyplot(bar_plot_all_Segments)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

            # Provide a download button for the chart image
                st.download_button("Download Segments", data=buffer, file_name="Segments.png")
            with st.expander("View Top Segemnts"):
                top_segments=prevalence
                top_segments['category'] = ''
                for key, words in result_dict.items():
                    top_segments.loc[top_segments.Segments.isin(words), 'category'] = key
                highest_rows = top_segments.groupby('category')['prevalence'].idxmax()
                top_segments = top_segments.loc[highest_rows].reset_index(drop=True).set_index('Segments')
                top_segments.rename(columns={'prevalence': 'Percentage'}, inplace=True)
                st.write(top_segments)

        with col3:
            select_all_segments = st.checkbox("Select All Segments to show the Plot -All categories combined",value=True)
        
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

        with col3:
            if select_all_segments:
                with st.expander("View All Segments in Plot Chart"):
                    scatter_plot=scatter_plot(merged_op)
                    st.pyplot(scatter_plot)
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    st.download_button("Download All Segments in Plot Chart-All categories combined", data=buffer, file_name="Plot Chart.png")
            else:
                selected_keys = st.multiselect("Select Category:", list(result_dict.keys()))
                # Create an empty list to store selected segments
                selected_segments = []

                # Iterate through selected keys and add corresponding segments to the selected_segments list
                for key in selected_keys:
                    selected_segments.extend(result_dict[key])

                merged_op=merged_op[selected_segments]
                with st.expander("View All Segments in Plot Chart"):
                    scatter_plot=scatter_plot(merged_op)
                    st.pyplot(scatter_plot)
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    st.download_button("Download All Segments in Plot Chart-All categories combined", data=buffer, file_name="Plot Chart.png")
            plt.title('Scatter Plot of Prevalence vs Overlap',color='white')
            plt.xlabel('Prevalence',color='white')
            plt.ylabel('Overlap',color='white')
            plt.legend()


        with col2:
            if oppotunity_options=='More popular & More Competition':
                with st.expander("View More Oppotunity- More overlap Area(From all category)"):
                    oppottunities('More popular & More Competition')
            elif oppotunity_options=='More popular & Less Competition':
                with st.expander("View More Oppotunity- Less overlap Area(From all category)"):
                    oppottunities("More popular & Less Competition")
            elif oppotunity_options=='Less popular & More Competition':
                with st.expander("View Less Oppotunity- More overlap Area(From all category)"):
                    oppottunities("Less popular & More Competition")
            else:
                with st.expander("View Less Oppotunity- Less overlap Area(From all category)"):
                    oppottunities("Less popular & Less Competition")


        prevalence,Top_left,Top_right,Bottom_left,Bottom_right,merged_op=Plot_chart_method(df_filtered,selected_segments,result_dict,False)
        
        match={'More popular & More Competition':Top_right,'More popular & Less Competition':Bottom_right,
                'Less popular & More Competition':Top_left,'Less popular & Less Competition':Bottom_left}
        prevalence = prevalence.sort_values(by='prevalence', ascending=False)

        with col3:
            select_all_segments = st.checkbox("Select All Segments to show the Plot -category wise",value=True)
        
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

        with col3:
            if select_all_segments:
                with st.expander("View All Segments in Plot Chart"):
                    scatter_plot=scatter_plot(merged_op)
                    st.pyplot(scatter_plot)
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    st.download_button("Download All Segments in Plot Chart", data=buffer, file_name="Plot Chart.png")
            else:
                selected_keys = st.multiselect("Select Category:", list(result_dict.keys()))
                # Create an empty list to store selected segments
                selected_segments = []

                # Iterate through selected keys and add corresponding segments to the selected_segments list
                for key in selected_keys:
                    selected_segments.extend(result_dict[key])

                merged_op=merged_op[selected_segments]
                with st.expander("View All Segments in Plot Chart"):
                    scatter_plot=scatter_plot(merged_op)
                    st.pyplot(scatter_plot)
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    st.download_button("Download All Segments in Plot Chart", data=buffer, file_name="Plot Chart.png")
            plt.title('Scatter Plot of Prevalence vs Overlap',color='white')
            plt.xlabel('Prevalence',color='white')
            plt.ylabel('Overlap',color='white')
            plt.legend()


        with col1:
            if oppotunity_options=='More popular & More Competition':
                with st.expander("View More Oppotunity- More overlap Area1 (From within each categories)"):
                    oppottunities1('More popular & More Competition')
            elif oppotunity_options=='More popular & Less Competition':
                with st.expander("View More Oppotunity- Less overlap Area1 (From within each categories)"):
                    oppottunities1("More popular & Less Competition")
            elif oppotunity_options=='Less popular & More Competition':
                with st.expander("View Less Oppotunity- More overlap Area1 (From within each categories)"):
                    oppottunities1("Less popular & More Competition")
            else:
                with st.expander("View Less Oppotunity- Less overlap Area1 (From within each categories)"):
                    oppottunities1("Less popular & Less Competition")
            

        @st.cache_data 
        def network_graph_method(df_filtered,selected_columns):
            network_df=df_filtered[selected_columns]
            network_df=filter_condition(network_df,selected_segments)
            for col in network_df.columns:
                network_df[col] = network_df[col].apply(lambda x: ','.join([val for val in str(x).split(',') if val in selected_segments]))

            def concat_non_blank(row):
                return ','.join([val for val in row if val != ''])

        # Apply the custom function to each row
            network_df['concatenated'] = network_df.apply(concat_non_blank, axis=1)
            network_df.drop(['interests', 'brands_visited', 'place_categories', 'geobehaviour'],axis=1,inplace=True)
            def count_elements(s):
                return len(s.split(','))

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
            edges_df = pd.DataFrame(np.sort(edges_df.values, axis=1), columns=edges_df.columns)

            edges_df['Value']=1
            edges_df=edges_df.groupby(['Source','Target'],sort=False,as_index=False).sum()
            edges_df = edges_df.sort_values(by='Value', ascending=False)

            return edges_df
        

        with col3:
            with st.expander("View Pair wise Segments"):
                edges_df=network_graph_method(df_filtered,selected_columns)
                st.write('Pair Wise count',edges_df.set_index('Source'))
                csv=edges_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Pair Wise Data",data=csv, file_name="pair wise segemnts.csv")

                G = nx.from_pandas_edgelist(edges_df, source='Source', target='Target', edge_attr='Value', create_using=nx.Graph())
                
                weighted_degree_centrality = {}
                for node in G.nodes():
                    weighted_degree = sum(G[node][neighbor]['Value'] for neighbor in G.neighbors(node))
                    weighted_degree_centrality[node] = weighted_degree
                degree_df = pd.DataFrame(list(weighted_degree_centrality.items()), columns=['Segment', 'Connection Strength']).sort_values('Connection Strength',ascending=False)

                st.write('Connection Strength Segments',degree_df.set_index('Segment'))
                csv=degree_df.to_csv(index=True).encode('utf-8')
                st.download_button("Download Connection Strength Segments",data=csv, file_name="Connection Strength Segments.csv")
    
if __name__=='__main__':
    main()