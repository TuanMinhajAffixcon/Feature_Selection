import boto3
from pyathena import connect
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
from adjustText import adjust_text

# Configure AWS credentials (using environment variables or other methods)
# ...

# Create a Boto3 Athena client
athena_client = boto3.client('athena', region_name='ap-southeast-2')

# Create a connection to Athena using pyathena
conn = connect(aws_access_key_id='AKIA2ZITI36WNI35MI2Z',
               aws_secret_access_key='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',
               s3_staging_dir='s3://tuan-query-result-bucket/query results/',
               region_name='ap-southeast-2')

# Create a cursor to execute queries
mycursor = conn.cursor()

def main():
    st.title('AFFIX Product Explorer')
    option = st.sidebar.selectbox("Select inputs", ('industry', 'segments', 'code'))
    mycursor.execute("select industry,code,category,segment_name from db_tuan_test.online_segments")
    results = mycursor.fetchall()
    column_names = [desc[0] for desc in mycursor.description]
    df_seg = pd.DataFrame(results, columns=column_names)
    df_seg.category = df_seg.category.str.upper()



    if option=='industry':
        # st.subheader('give a industry')
        selected_industry = st.text_input("Enter Industry")
        segment_industry_dict = df_seg.groupby('industry')['segment_name'].apply(list).to_dict()
        if selected_industry in segment_industry_dict:
            item_list = segment_industry_dict[selected_industry]

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

    elif option == 'segments':
        st.subheader('give segments as comma separated values')
        selected_segments = st.text_input("Enter segments")
        item_list=[selected_segments]
        item_list=[x for xs in item_list for x in xs.split(',')]
    elif option == 'code':
        st.subheader('give a code')
        selected_code = st.text_input("Enter code")
        item_list = []
        segment_industry_dict = df_seg.groupby('code')['segment_name'].apply(list).to_dict()
        user_contain_list = list(df_seg[['code']][df_seg.code.str.contains(selected_code)].code.unique())
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
        # If the value is not already a key in result_dict, add it and initialize with an empty list
        if value not in result_dict:
            result_dict[value] = []
        # Append the key to the list corresponding to the value in result_dict
        result_dict[value].append(key)
        result_dict = {key: values for key, values in result_dict.items()}
    # result_dict = {key: result_dict[key] for key in ['INTERESTS','BRAND VISITED','PLACE CATEGORIES','GEO BEHAVIOUR']}
    # selected_category = st.sidebar.selectbox("Select Category", ('INTERESTS', 'BRAND VISITED', 'PLACE CATEGORIES','GEO BEHAVIOUR'))
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


    # item_list = selected_segments
    
    # st.write(segment_list)
    for j in segment_list:
        st.sidebar.write(j)
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    selected_columns = ['interests', 'brands_visited', 'place_categories', 'geobehaviour']
    def filter_condition(df,lst):

        filter_conditions = [df[col_name].apply(lambda x: any(item in str(x).split(',') for item in lst))
            for col_name in selected_columns]

        #
        # # Combine filter conditions using OR operator
        final_condition = filter_conditions[0]
        for condition in filter_conditions[1:]:
            final_condition = final_condition | condition

        # # Apply the filter to the DataFrame
        df_new = df[final_condition]
        return df_new
    
    if uploaded_file is not None:
        # Load data from the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        
        st.write("Your uploaded data count is: ",len(df))

        df=filter_condition(df,selected_segments)
        
        st.write("Your Matched data count is: ",len(df))
        
        bin_edges = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,  float('inf')]  # Define your age bins here
        bin_labels = ['less than 20', 'Age 20-24', 'Age 25-29','Age 30-34', 'Age 35-39','Age 40-44', 'Age 45-49','Age 50-54', 'Age 55-59','Age 60-64', 'Age 65-69','Age 70-74', 'Age 75-79','Age 80-84','85+']  
        # Use cut to create age groups based on bin edges and assign unknown values to 'unknown_age'
        df['age_category'] = pd.cut(df['age'].fillna(-1), bins=bin_edges, labels=bin_labels, include_lowest=True)

        # Replace the 'unknown_age' category with NaN for rows where 'age' is NaN
        df['age_category'] = df['age_category'].cat.add_categories('unknown_age').fillna('unknown_age')

        gender_mapping = {'female': 'Female', 'male': 'Male'}
        # Replace values in the 'gender' column using the mapping
        df['gender'] = df['gender'].replace(gender_mapping)

        
        df['gender'].fillna('unknown_gender',inplace=True)
        df['income'].fillna('unknown_income',inplace=True)
        
        
        
        # df.drop(['maid','year','month'],axis=1,inplace=True)
        from sklearn.preprocessing import OneHotEncoder
        obj=OneHotEncoder()
        demographics_df=df[['income','gender','age_category']]
        def demographics(df):
            df=obj.fit_transform(df).toarray()

            # Get the feature names for the one-hot encoded columns
            encoded_feature_names = obj.get_feature_names_out(input_features=['income', 'gender', 'age_category'])

            # Create a DataFrame with the one-hot encoded columns and set column names
            df = pd.DataFrame(df, columns=encoded_feature_names)

            def remove_words_from_column_names(col_name):
                words_to_remove = ['income_', 'gender_', 'age_category_']
                for word in words_to_remove:
                    col_name = col_name.replace(word, '')
                return col_name

            # Rename columns by applying the remove_words_from_column_names function
            df.rename(columns=remove_words_from_column_names, inplace=True)

            # Calculate the column sums
            column_sums = df.sum()

            # Calculate the percentage for each column
            percentages = (column_sums / len(df)) * 100

            # # Create a new DataFrame with column names and percentages
            df = pd.DataFrame({'Column Name': df.columns, 'Percentage': percentages})

            return df
        
        demographics_df=demographics(demographics_df)
        # st.write(demographics_df)

        def bar_chart(df):

            plt.figure(figsize=(15, 9))
            colors=['orange']*8+['green']*3+['blue']*16
            bars = plt.bar(df['Column Name'], df['Percentage'],color=colors)
            plt.xlabel('Demographics')
            plt.ylabel('Percentage (%)')
            plt.title('Percentage of Each Column')
            plt.xticks(rotation=90)

            for bar, value in zip(bars, df['Percentage']):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom',rotation=90)
            return plt
        bar_chart(demographics_df)

        plt.tight_layout()  
        st.pyplot(plt)

        select_all_segments=st.sidebar.selectbox('Selection Demographics',('Select All Demographics','Custom Demographics'))
        # select_all_segments=
        
        if select_all_segments=='Select All Demographics':
            df_filtered=df
    
        else:
            selected_age_category = st.sidebar.multiselect("Select Age Category", ('less than 20', 'Age 20-24', 'Age 25-29','Age 30-34', 'Age 35-39','Age 40-44', 'Age 45-49','Age 50-54', 'Age 55-59','Age 60-64', 'Age 65-69','Age 70-74', 'Age 75-79','Age 80-84','85+'))
            selected_gender_category = st.sidebar.multiselect("Select Gender Category", ('Male','Female'))
            selected_income_category = st.sidebar.multiselect("Select Income Category", ("Very Low (Under $20,799)", "Low ($20,800 - $41,599)", "Below Average ($41,600 - $64,999)","Average ($65,000 - $77,999)", "Above Average ($78,000 - $103,999)","High ($104,000 - $155,999)","Very high ($156,000+)"))

            filter_criteria=selected_age_category+selected_gender_category+selected_income_category
            df_filtered = df[df.isin(filter_criteria).any(axis=1)]


        st.write(len(df_filtered))
#-----------------------------------------------------------------------------------------------------------------------------------

        segments_filtered_df=df_filtered[selected_columns]
        
        for item in selected_segments:
            segments_filtered_df[item] = segments_filtered_df.apply(lambda row: 1 if any(
                str(item) in val.split(',') if isinstance(val, str) else False for val in row) else 0, axis=1)
        segments_filtered_df.drop(selected_columns,axis=1,inplace=True)
        
        new_order = ["BRAND VISITED", "PLACE CATEGORIES", "INTERESTS", "GEO BEHAVIOUR"]

        # Create a list of items
        items = list(result_dict.keys())

        # Sort the list based on the custom order
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
        # prevalence.index=

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
        # st.write(df_quadrant)

        select_all_segments = st.checkbox("Select All Segments to show the Plot")
        def scatter_plot(df):
            plt.figure(figsize=(15, 12))  # Set the figure size
            # fig, ax = plt.subplots(figsize=(15, 12))
            #
            # # # Scatter plot
            scatter=plt.scatter(df.loc['prevalence'], df.loc['overlap'], color='blue', marker='o', label='All Segments')
            #
            # # # Annotate points with words
            for x, y, word in zip(df.loc['prevalence'], df.loc['overlap'], df.columns):
                plt.annotate(word, (x, y), textcoords="offset points",xytext=(10,-200), ha='center',
                             arrowprops=dict(facecolor='red',edgecolor='blue',arrowstyle='->',connectionstyle='angle'))

            # # Highlight x and y axes
            plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Highlight x axis
            plt.axvline(0, color='black', linewidth=1, linestyle='--')  # Highlight y axis
            return plt

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
        plt.title('Scatter Plot of Prevalence vs Overlap')
        plt.xlabel('Prevalence')
        plt.ylabel('Overlap')
        plt.legend()
        st.pyplot(plt.gcf())
        

#----------------------------------------------------------------------------------------------------------------------

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
            fig, ax1 = plt.subplots(figsize=(10, 9))
            colors=['orange']*8+['green']*3+['blue']*16
            bars = ax1.bar(bar_df['Column Name'], bar_df['Percentage'], color=colors, alpha=0.7, label='df_demographics_Top3 (Bar Chart)')
            ax1.set_xlabel('Demographics')
            ax1.set_ylabel('Segment- Percentage', color='brown')
            ax1.set_title('Profiling')
            plt.xticks(rotation=90)
            for bar, value in zip(bars, bar_df['Percentage']):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom',
                        color='brown')
    #
            # # # Create a twin right y-axis for df2
            ax2 = ax1.twinx()
            line = ax2.plot(bar_df['Column Name'], index_df['index'], marker='o', linestyle='-', color='red', label='index (Line Chart)')
            ax2.set_ylabel('index', color='red')
            #
            # # Add data labels for the points in the line chart (df2)
            for x, y in zip(bar_df['Column Name'], index_df['index']):
                label = f'{y:.2f}'  # Format the label as desired
                ax2.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='red',rotation=90)


            # Combine legends from both axes
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()

            return plt,ax1,ax2
        selected_segment = st.selectbox("Segmentation Selection:", ('Top_3_Segments_in_each_category', 'Custom_Selection'))
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

        else:
            selected_segments = st.multiselect("Select segments:", segment_list)
            if len(selected_segments)>0:
                custom_df=df_filtered
                custom_df =filter_condition(custom_df,selected_segments)
                custom_df=custom_df[['income','gender','age_category']]
                custom_df=demographics(custom_df)

                index=pd.merge(custom_df,demographics_df,on='Column Name',how='inner')
                
                index['index'] = (index.iloc[:,1] / index.iloc[:,2]*100)
                twin_bar_chart(custom_df,index)
                plt.tight_layout()

                st.pyplot(plt)
            else:
                selected_segments=[]

#----------------------------------------------------------------------------------------------------------------------------------

        network_df = Quadrant_selection[selected_columns]

        def concatenate_without_nan(row):
            return ','.join([str(value) for value in row if not pd.isna(value)])

        # Apply the function to each row and store the result in a new column 'concatenated'
        network_df['concatenated'] = network_df.apply(concatenate_without_nan, axis=1)
        network_df.drop(columns=selected_columns, inplace=True)

        def concat_matching_items(column, item_list):
            return ','.join([item for item in item_list if column and item in column])
        #Apply the custom function to each column
        network_df['concatenated'] = network_df['concatenated'].apply(lambda x: concat_matching_items(x, segments_list))

        # Function to count elements after splitting
        def count_elements(s):
            return len(s.split(','))

        # Filter rows where the 'interests' column has more than one element
        network_df = network_df[network_df['concatenated'].apply(count_elements) > 1]

        network_df['concatenated_items'] = network_df['concatenated'].str.split(',')
        nodes_df = pd.DataFrame(network_df['concatenated_items'].explode().unique(), columns=['Node'])
        edges = []

        for _, row in network_df.iterrows():
            nodes = row['concatenated_items']
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    edges.append((nodes[i], nodes[j]))

        edges_df = pd.DataFrame(edges, columns=['Source', 'Target'])
        # Create a Pyvis network
        network = Network(notebook=True, width='800px', height='600px')

        # Add nodes
        for _, node in nodes_df.iterrows():
            network.add_node(node['Node'])

        # Add edges
        for _, edge in edges_df.iterrows():
            network.add_edge(edge['Source'], edge['Target'])

        edges_df = pd.DataFrame(np.sort(edges_df.values, axis=1), columns=edges_df.columns)

        edges_df['Value']=1
        edges_df=edges_df.groupby(['Source','Target'],sort=False,as_index=False).sum()
        edges_df = edges_df.sort_values(by='Value', ascending=False)

        # Create a NetworkX graph
        G = nx.from_pandas_edgelist(edges_df, source='Source', target='Target', edge_attr='Value', create_using=nx.Graph())

        # Create a Pyvis Network graph
        net = Network(notebook=True, width='1000px', height='700px', bgcolor='#222222', font_color='white')

        # Set attributes
        node_degree = dict(G.degree)
        communities = community_louvain.best_partition(G)
        nx.set_node_attributes(G, node_degree, 'size')
        nx.set_node_attributes(G, communities, 'group')

        # Convert to Pyvis
        net.from_nx(G)

        # Export to HTML
        # net.show('com_net_network_graph.html')

        # Display in Streamlit
        # st.components.v1.html(open("com_net_network_graph.html").read(), height=700, width=1000)

                # Convert the dictionary to a DataFrame
        communities_df = pd.DataFrame.from_dict(communities, orient='index', columns=['Community'])

        # Reset the index to have keys as rows
        communities_df.reset_index(inplace=True)
        communities_df.rename(columns={'index': 'Key'}, inplace=True)

        # Sort the DataFrame by the "Community" column in ascending order
        communities_df.sort_values(by='Community', ascending=True, inplace=True)

        # Reset the index after sorting
        communities_df.reset_index(drop=True, inplace=True)

    #-----------------------------------------------------------------------------------------------------------------------------

        degree_dict=nx.degree_centrality(G)
        degree_df=pd.DataFrame.from_dict(degree_dict,orient="index",columns=['centrality']).sort_values(by='centrality', ascending=False)
        
        def bar_chart_centrality(df,y_label,title,bar_color):
            plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

            # Use barplot from Matplotlib to create the bar chart
            plt.barh(df.index, df['centrality'], color=bar_color)

            # Add labels and title
            plt.xlabel('Segments')
            plt.ylabel(y_label)
            plt.title(title)

            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=90)

            # Show the plot
            plt.tight_layout()  # Ensures labels fit within the figure
            plt.show()

            return plt

        degree_df=bar_chart_centrality(degree_df,'Degree Centrality Value','Degree Centrality of Segments (represent how many charactors)','orange')

        plt.tight_layout()  
        st.pyplot(plt)

        betweenness_dict=nx.betweenness_centrality(G)
        betweenness_df=pd.DataFrame.from_dict(betweenness_dict,orient="index",columns=['centrality']).sort_values(by='centrality', ascending=False)
        betweenness_df=bar_chart_centrality(betweenness_df,'Betweenness Centrality Value','Betweenness Centrality of Segments (Represent how connects communities)','green')

        plt.tight_layout()  
        st.pyplot(plt)

        clossness_dict=nx.closeness_centrality(G)
        betweenness_df=pd.DataFrame.from_dict(clossness_dict,orient="index",columns=['centrality']).sort_values(by='centrality', ascending=False)
        betweenness_df=bar_chart_centrality(betweenness_df,'Clossness Centrality Value','Clossness Centrality of Segments (Represent segments that are the closest)','purple')

        plt.tight_layout()  
        st.pyplot(plt)


        nx.set_node_attributes(G,degree_dict,'degree_centrality')
        nx.set_node_attributes(G,betweenness_dict,'betweenness_centrality')
        nx.set_node_attributes(G,clossness_dict,'clossness_centrality')

        communities = community_louvain.best_partition(G)
        # nx.set_node_attributes(G, node_degree, 'size')
        nx.set_node_attributes(G, communities, 'group')

        com_net = Network(notebook=True, width='1000px', height='700px', bgcolor='#222222', font_color='white')

        com_net.from_nx(G)

        # Export to HTML
        com_net.show('com_net_network_graph.html')

        # Display in Streamlit
        st.components.v1.html(open("com_net_network_graph.html").read(), height=700, width=1000)
        

        


        

        

        # st.write(betweenness_dict)
        st.write(edges_df)
        st.write(communities_df)










if __name__=='__main__':
    main()
