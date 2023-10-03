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
    mycursor.execute("select * from db_tuan_test.online_segments")
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
    filtered_dict = {key: value for key, value in segment_category_dict.items() if key in item_list}

    for key, value in filtered_dict.items():
        # If the value is not already a key in result_dict, add it and initialize with an empty list
        if value not in result_dict:
            result_dict[value] = []
        # Append the key to the list corresponding to the value in result_dict
        result_dict[value].append(key)
        result_dict = {key: values for key, values in result_dict.items()}
    selected_category = st.sidebar.selectbox("Select Category", ('INTERESTS', 'BRAND VISITED', 'PLACE CATEGORIES','GEO BEHAVIOUR'))
    if item_list:
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
    # for i in selected_segments:
        # st.write(i)


    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load data from the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        # st.write("")
        st.write("Your uploaded data count is: ",len(df))
        selected_columns = ['interests', 'brands_visited', 'place_categories', 'geobehaviour']
        def map_age_to_category(age):
            if 20 <= age <= 24:
                return "Age 20-24"
            elif 25 <= age <= 29:
                return "Age 25-29"
            elif 30 <= age <= 34:
                return "Age 30-34"
            elif 35 <= age <= 39:
                return "Age 25-29"
            elif 40 <= age <= 44:
                return "Age 30-34"
            elif 45 <= age <= 49:
                return "Age 25-29"
            elif 50 <= age <= 54:
                return "Age 30-34"
            elif 55 <= age <= 59:
                return "Age 30-34"
            elif 60 <= age <= 64:
                return "Age 25-29"
            elif 65 <= age <= 69:
                return "Age 30-34"
            elif 70 <= age <= 74:
                return "Age 25-29"
            elif 75 <= age <= 79:
                return "Age 30-34"
            elif 80 <= age <= 84:
                return "Age 25-29"
            elif age > 84:
                return ">84"
            else:
                return "Unknown"

        df["age_category"] = df["age"].apply(map_age_to_category)

        filter_conditions = [
            df[col_name].apply(lambda x: isinstance(x, str) and any(item in x.split(',') for item in selected_segments))
            for col_name in selected_columns
        ]

        # Combine the filter conditions using logical OR
        final_condition = filter_conditions[0]
        for condition in filter_conditions[1:]:
            final_condition = final_condition | condition

        # Apply the final filter
        filtered_df = df[final_condition]

        st.write("Your matched data count is: ", len(filtered_df))
        filtered_df_overall=filtered_df
        # st.write(filtered_df_overall.columns)
        filtered_df = filtered_df[['maid']+selected_columns]

        for column in selected_columns:
            if column in filtered_df.columns:
                filtered_df[column] = filtered_df[column].apply(
                    lambda x: ','.join([item for item in selected_segments if isinstance(x, str) and item in x.split(',')]))


        for item in selected_segments:
            filtered_df[item] = filtered_df.apply(lambda row: 1 if any(
                str(item) in val.split(',') if isinstance(val, str) else False for val in row) else 0, axis=1)


        df_demographics=df[['maid','gender','income','age_category']]
        df_demographics['concatinated_column'] = df_demographics['gender'] + '_' + df_demographics['income'] + '_' + df_demographics['age_category']


        age_groups = ["Age 20-24", "Age 25-29", "Age 30-34", "Age 35-39", "Age 40-44", "Age 45-49", "Age 50-54",
                      "Age 55-59", "Age 60-64", "Age 65-69", "Age 70-74", "Age 75-79", "Age 80-84", "Age 85+"]
        gender_groups = ["male", "female"]
        income_groups = ["Very Low (Under $20,799)", "Low ($20,800 - $41,599)", "Below Average ($41,600 - $64,999)",
                         "Average ($65,000 - $77,999)", "Above Average ($78,000 - $103,999)",
                         "High ($104,000 - $155,999)",
                         "Very high ($156,000+)"]
        all_groups = age_groups + gender_groups + income_groups
        df_demographics = df_demographics.assign(**{col: None for col in all_groups})

        for item in all_groups:
            df_demographics[item] = df_demographics.apply(lambda row: 1 if any(
                str(item) in val.split('_') if isinstance(val, str) else False for val in row) else 0, axis=1)
        df_demographics_overall = df_demographics[df_demographics['maid'].isin(filtered_df['maid'])]
        df_demographics_overall=df_demographics_overall[df_demographics_overall.columns[5:]]

        column_counts = {'Total Count': len(df_demographics_overall)}

        # Calculate counts for each count_column
        for col_name in df_demographics_overall.columns:
            column_counts[f"{col_name} Count"] = df_demographics_overall[col_name].sum()

        # Create a DataFrame from the counts dictionary
        count_df  = pd.DataFrame(column_counts, index=[0])

        # Calculate percentages and round them to two decimal places

        for col_name in df_demographics_overall.columns:
            count_df [f"{col_name}"] = round(count_df [f"{col_name} Count"] / count_df ['Total Count'] * 100, 2)
        df_demographics_overall=count_df [count_df .columns[24:]]

        # st.dataframe((df_demographics_overall))

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
        new_columns_order = result_dict['BRAND VISITED'] + result_dict['PLACE CATEGORIES'] \
                            + result_dict['INTERESTS'] + result_dict['GEO BEHAVIOUR']

        # Reorder the columns of the DataFrame
        filtered_df_avg = filtered_df[new_columns_order]

        filtered_df = filtered_df.drop(columns=selected_columns, axis=1)
        filtered_df_Top3 = filtered_df
        selected_segments_df=filtered_df
        filtered_df = filtered_df[new_columns_order]
        for key, columns in result_dict.items():
            #     print(columns)
            for index, row in filtered_df_avg.iterrows():
                # Calculate the sum and count for the current set of columns
                row_sum = row[columns].sum()
                row_count = row[columns].count()
                value = round((row_sum / row_count), 2)

                # Replace values of 1 with the calculated value, leave 0 as it is
                filtered_df_avg.loc[index, columns] = [value if x == 1 else x for x in row[columns]]
        prevalence = filtered_df.agg(lambda col: round(col.mean() * 100, 2)).transpose()

        test = prevalence
        # st.write(prevalence)

        # Calculate sum of non-zero values for each column
        sum_non_zero = filtered_df_avg.apply(lambda col: col[col != 0].sum())

        # Calculate count of non-zero values for each column
        count_non_zero = filtered_df_avg.apply(lambda col: (col != 0).count())

        # Calculate the ratio by dividing sum by count
        overlap = round((sum_non_zero / count_non_zero)*100,2).transpose()
        # st.write(overlap)
        merged_op = pd.concat([prevalence, overlap],axis=1).transpose()
        merged_op.index = ['prevalence', 'overlap']
        for new_col, original_cols in result_dict.items():
            merged_op[new_col + '_avg'] = merged_op[original_cols].mean(axis=1)
            merged_op[new_col + '_std'] = merged_op[original_cols].std(axis=1,ddof=0)
        # st.write(merged_op)
        new_row_values = []
        for _, row in merged_op.iterrows():
            new_row = []
            for key, original_cols in result_dict.items():
                new_row.extend((row[original_cols] - row[key + '_avg']) / row[key + '_std'])
            new_row_values.append(new_row)

        df_new_row = pd.DataFrame(new_row_values, columns=merged_op.columns[:-len(result_dict) * 2])
        # st.write(overlap)
        merged_op = df_new_row
        merged_op = merged_op.fillna(0)
        merged_op.index = merged_op.index.map({0: 'prevalence', 1: 'overlap'})

        df_quadrant = merged_op.transpose()
        df_quadrant['quadrant'] = 'None'
        df_quadrant.loc[(df_quadrant['prevalence'] > 0) & (df_quadrant['overlap'] >= 0), 'quadrant'] = 'Top_right'
        df_quadrant.loc[(df_quadrant['prevalence'] <= 0) & (df_quadrant['overlap'] >= 0), 'quadrant'] = 'Top_left'
        df_quadrant.loc[(df_quadrant['prevalence'] <= 0) & (df_quadrant['overlap'] < 0), 'quadrant'] = 'Bottom_left'
        df_quadrant.loc[(df_quadrant['prevalence'] > 0) & (df_quadrant['overlap'] < 0), 'quadrant'] = 'Bottom_right'


        # Calculate the distance from the origin (0, 0) for each point
        distances = np.sqrt(df_quadrant['prevalence'] ** 2 + df_quadrant['overlap'] ** 2)

        # Calculate the distance from the origin (100, -100) for each point
        distances_from_100 = np.sqrt((100 - df_quadrant['prevalence']) ** 2 + (-100 - df_quadrant['overlap']) ** 2)

        # Calculate the angle for each point
        angles = np.arctan2(df_quadrant['overlap'], df_quadrant['prevalence'])

        # Convert angles from radians to degrees
        angles_degrees = np.degrees(angles)

        # Calculate the acute absolute angle from the horizontal axis
        angles_from_horizontal = np.abs(90 - np.abs(angles_degrees))

        # Handle the case where x and y are both zero
        angles_from_horizontal[(df_quadrant['prevalence'] == 0) & (df_quadrant['overlap'] == 0)] = 0

        # Create a new column with updated values based on conditions
        df_quadrant['updated_angle_degrees'] = np.where(angles_from_horizontal > 45, 45 - (angles_from_horizontal - 45),
                                                        angles_from_horizontal)

        # Add a new column for distances from origin
        df_quadrant['distance_from_origin'] = distances
        df_quadrant['distance_from_(100,-100)'] = distances_from_100

        # Create the 'Interim_score_Bottom_Right_Quadrant' column
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
        interim_df=sorted_df[['quadrant', 'interim_score', 'Overall Ranking Within Quadrant']]

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


        select_all_segments = st.checkbox("Select All Segments to show the Plot")
        if select_all_segments:
            # Plotting
            plt.figure(figsize=(15, 12))  # Set the figure size
            # fig, ax = plt.subplots(figsize=(15, 12))
            #
            # # # Scatter plot
            scatter=plt.scatter(merged_op.loc['prevalence'], merged_op.loc['overlap'], color='blue', marker='o', label='All Segments')
            #
            # # # Annotate points with words
            for x, y, word in zip(merged_op.loc['prevalence'], merged_op.loc['overlap'], merged_op.columns):
                plt.annotate(word, (x, y), textcoords="offset points",xytext=(10,-200), ha='center',
                             arrowprops=dict(facecolor='red',edgecolor='blue',arrowstyle='->',connectionstyle='angle'))

            # # Highlight x and y axes
            plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Highlight x axis
            plt.axvline(0, color='black', linewidth=1, linestyle='--')  # Highlight y axis
        else:
            selected_keys = st.multiselect("Select Category:", list(result_dict.keys()))
            # Create an empty list to store selected segments
            selected_segments = []

            # Iterate through selected keys and add corresponding segments to the selected_segments list
            for key in selected_keys:
                selected_segments.extend(result_dict[key])

            merged_op=merged_op[selected_segments]

            plt.figure(figsize=(15, 12))  # Set the figure size
            # fig, ax = plt.subplots(figsize=(15, 12))
            #
            # # # Scatter plot
            scatter = plt.scatter(merged_op.loc['prevalence'], merged_op.loc['overlap'], color='blue', marker='o',
                                  label=selected_keys)
            #
            # # # Annotate points with words
            for x, y, word in zip(merged_op.loc['prevalence'], merged_op.loc['overlap'], merged_op.columns):
                plt.annotate(word, (x, y), textcoords="offset points",xytext=(10,-200), ha='center',
                             arrowprops=dict(facecolor='red', edgecolor='blue', arrowstyle='->',
                                             connectionstyle='angle'))

            # # Highlight x and y axes
            plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Highlight x axis
            plt.axvline(0, color='black', linewidth=1, linestyle='--')  # Highlight y axis




        plt.title('Scatter Plot of Prevalence vs Overlap')
        plt.xlabel('Prevalence')
        plt.ylabel('Overlap')
        plt.legend()
        st.pyplot(plt.gcf())
        # st.write("You selected segments:", selected_segments)
        # st.dataframe(merged_op)
        # st.dataframe(filtered_df)
        selected_quadrant = st.selectbox("Select Quadrant:", ('Top_right', 'Bottom_right', 'Top_left', 'Bottom_left'))
        if "Top_right" in selected_quadrant:
            selected_quadrant = Top_right
            segments_list = selected_quadrant.index.tolist()


        elif "Bottom_right" in selected_quadrant:
            selected_quadrant = Bottom_right
            segments_list = selected_quadrant.index.tolist()

        elif "Top_left" in selected_quadrant:
            selected_quadrant = Top_left
            segments_list = selected_quadrant.index.tolist()

        elif "Bottom_left" in selected_quadrant:
            selected_quadrant = Bottom_left
            segments_list = selected_quadrant.index.tolist()


        # Create filter conditions for each selected column
        # filtered_df_overall=filtered_df_overall[selected_columns]
        # Create filter conditions for each selected column
        filter_conditions = [
            filtered_df_overall[col_name].apply(lambda x: any(item in str(x).split(',') for item in segments_list))
            for col_name in selected_columns]

        #
        # # Combine filter conditions using OR operator
        final_condition = filter_conditions[0]
        for condition in filter_conditions[1:]:
            final_condition = final_condition | condition

        # # Apply the filter to the DataFrame
        Quadrant_maid = filtered_df_overall[final_condition]

        for column in selected_columns:
            if column in Quadrant_maid.columns:
                Quadrant_maid[column] = Quadrant_maid[column].apply(
                    lambda x: ','.join([item for item in segments_list if isinstance(x, str) and item in x.split(',')]))
        #
        #

        # st.write(segments_list)
        # st.write(len(Quadrant_maid))
        overall_percentage = (len(Quadrant_maid) / len(filtered_df_overall)) * 100
        st.write(f"Showing Data Percentage :",f"{overall_percentage:.2f}%")
        st.write(selected_quadrant)

        network_df = Quadrant_maid[selected_columns]

        # Function to concatenate non-blank items in a row
        def concatenate_non_blank(row):
            return ','.join([item for item in row if item != ''])

        # Apply the function to each row and store the result in a new column 'concatenated'
        network_df['concatenated'] = network_df.apply(concatenate_non_blank, axis=1)
        network_df.drop(columns=selected_columns, inplace=True)
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

        G=nx.from_pandas_edgelist(edges_df,source='Source',target='Target',edge_attr='Value',create_using=nx.Graph())
        plt.figure(figsize=(10,10))
        pos=nx.kamada_kawai_layout(G)
        nx.draw(G,with_labels=True,node_color='skyblue',edge_cmap=plt.cm.Blues,pos=pos)
        # st.pyplot(plt)
        net = Network(notebook=True, width='1000px', height='700px', bgcolor='#222222', font_color='white')
        node_degree = dict(G.degree)
        communities = community_louvain.best_partition((G))
        nx.set_node_attributes(G, node_degree, 'size')
        # st.write(communities)
        nx.set_node_attributes(G, communities, 'group')
        com_net = Network(notebook=True, width='1000px', height='700px', bgcolor='#222222', font_color='white')
        com_net.from_nx(G)
        com_net.repulsion(node_distance=9000,spring_length=20000)

        com_net.set_options("""
                var options = {
                  "physics": {
                    "enabled": false
                  }
                }
                """)
        com_net.show('com_net_network_graph.html')

        st.components.v1.html(open("com_net_network_graph.html").read(), height=700, width=1000)

        st.write(nodes_df)
        st.write(edges_df)





        selected_quadrant = selected_quadrant.reset_index()
        selected_quadrant = selected_quadrant.rename(columns={selected_quadrant.columns[0]: 'segments'})

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

            filtered_df_Top3=filtered_df_Top3[['maid']+concatenated_list]
            filtered_df_Top3 = filtered_df_Top3.loc[~filtered_df_Top3.iloc[:, 1:].eq(0).all(axis=1)]
            df_demographics_Top3 = df_demographics[df_demographics['maid'].isin(filtered_df_Top3['maid'])]
            df_demographics_Top3 = df_demographics_Top3[df_demographics_Top3.columns[5:]]
            # # Initialize a dictionary to store column counts
            column_counts = {'Total Count': len(df_demographics_Top3)}

            # Calculate counts for each count_column
            for col_name in df_demographics_Top3.columns:
                column_counts[f"{col_name} Count"] = df_demographics_Top3[col_name].sum()

            # Create a DataFrame from the counts dictionary
            count_df = pd.DataFrame(column_counts, index=[0])

            # Calculate percentages and round them to two decimal places

            for col_name in df_demographics_overall.columns:
                count_df[f"{col_name}"] = round(count_df[f"{col_name} Count"] / count_df['Total Count'] * 100, 2)

            df_demographics_Top3 = count_df[count_df.columns[24:]]
            index = pd.concat([df_demographics_overall, df_demographics_Top3])
            # Calculate the division result for the second row by the first row
            # # Calculate the division result for the second row by the first row
            division_result = index.iloc[1] / index.iloc[0]*100

            # # Create a new DataFrame with the division result
            new_row = pd.DataFrame([division_result], columns=index.columns).fillna(0)

            # # Append the new row to the original DataFrame
            index = pd.concat([index, new_row]).fillna(0)
            new_index_names = ['Overall', 'Top3', 'index']

            index.index = new_index_names
            index=index.transpose().reset_index()
            index.columns = ['Demographics','Overall', 'Top3', 'index']
            index=index[['Demographics','index']]

            df_demographics_Top3 = df_demographics_Top3.transpose()

            # df_demographics_Top3.index.name = 'Demographics'
            df_demographics_Top3 = df_demographics_Top3.reset_index()
            df_demographics_Top3.columns = ['Demographics', 'Percentage']

            # # Create a figure and left y-axis for df1
            fig, ax1 = plt.subplots(figsize=(10, 9))
            bars = ax1.bar(df_demographics_Top3['Demographics'], df_demographics_Top3['Percentage'], color='blue', alpha=0.7, label='df_demographics_Top3 (Bar Chart)')
            ax1.set_xlabel('Demographics')
            ax1.set_ylabel('Segment- Percentage', color='blue')
            ax1.set_title('Profiling')
            plt.xticks(rotation=90)
            for bar, value in zip(bars, df_demographics_Top3['Percentage']):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom',
                         color='blue')
            #
            # # Create a twin right y-axis for df2
            ax2 = ax1.twinx()
            line = ax2.plot(index['Demographics'], index['index'], marker='o', linestyle='-', color='red', label='index (Line Chart)')
            ax2.set_ylabel('index', color='red')
            #
            # # Add data labels for the points in the line chart (df2)
            for x, y in zip(index['Demographics'], index['index']):
                label = f'{y:.2f}'  # Format the label as desired
                ax2.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='red',rotation=90)


            # Combine legends from both axes
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # ax1.legend(lines + lines2, labels + labels2, loc='upper right')


            # plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(plt)
            # st.write("Top Ranking Segments:", top_segments_by_category)
            # top_segments_by_category=pd.DataFrame(top_segments_by_category)
            # st.dataframe(top_segments_by_category)
            # filtered_dict = {key: [value for value in values if value in selected_segments] for key, values in top_segments_by_category.items()}
            # filtered_data = {key: value for key, value in filtered_dict.items() if (value and (isinstance(value, list) and len(value) > 0))}
            for key, values in top_segments_by_category.items():
                df = pd.DataFrame({key: values})
                st.write(df)




        else:
                # Create an empty dictionary to store the top three segments in each category
                selected_segments = st.multiselect("Select segments:", item_list)

                # st.write(selected_segments)

                selected_segments_df = selected_segments_df[['maid'] + selected_segments]
                selected_segments_df = selected_segments_df.loc[~selected_segments_df.iloc[:, 1:].eq(0).all(axis=1)]
                selected_segments_df = df_demographics[df_demographics['maid'].isin(selected_segments_df['maid'])]
                selected_segments_df = selected_segments_df[selected_segments_df.columns[5:]]
                # # # Initialize a dictionary to store column counts
                column_counts = {'Total Count': len(selected_segments_df)}
                #
                # # Calculate counts for each count_column
                for col_name in selected_segments_df.columns:
                    column_counts[f"{col_name} Count"] = selected_segments_df[col_name].sum()
                #
                # # Create a DataFrame from the counts dictionary
                count_df = pd.DataFrame(column_counts, index=[0])
                #
                # # Calculate percentages and round them to two decimal places
                #
                for col_name in df_demographics_overall.columns:
                    count_df[f"{col_name}"] = round(count_df[f"{col_name} Count"] / count_df['Total Count'] * 100, 2)

                selected_segments_df = count_df[count_df.columns[24:]]

                index = pd.concat([df_demographics_overall, selected_segments_df])
                # # Calculate the division result for the second row by the first row
                # # # Calculate the division result for the second row by the first row
                division_result = index.iloc[1] / index.iloc[0] * 100
                #
                # # # Create a new DataFrame with the division result
                new_row = pd.DataFrame([division_result], columns=index.columns).fillna(0)
                #
                # # # Append the new row to the original DataFrame
                index = pd.concat([index, new_row]).fillna(0)
                new_index_names = ['Overall', 'Selected_Segments', 'index']
                #
                index.index = new_index_names
                index = index.transpose().reset_index()
                index.columns = ['Demographics', 'Overall', 'Selected_Segments', 'index']
                index = index[['Demographics', 'index']]
                #
                selected_segments_df = selected_segments_df.transpose()

                selected_segments_df.index.name = 'Demographics'
                selected_segments_df = selected_segments_df.reset_index()
                selected_segments_df.columns = ['Demographics', 'Percentage']
                # st.dataframe(index)
                # st.dataframe(selected_segments_df)
                # # # Create a figure and left y-axis for df1
                fig, ax1 = plt.subplots(figsize=(10, 9))
                bars = ax1.bar(selected_segments_df['Demographics'], selected_segments_df['Percentage'], color='blue',
                               alpha=0.7, label='selected_segments (Bar Chart)')
                ax1.set_xlabel('Demographics')
                ax1.set_ylabel('Segment- Percentage', color='blue')
                ax1.set_title('Profiling')
                plt.xticks(rotation=90)
                for bar, value in zip(bars, selected_segments_df['Percentage']):
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center',
                             va='bottom',
                             color='blue')
                # #
                # # # Create a twin right y-axis for df2
                ax2 = ax1.twinx()
                line = ax2.plot(index['Demographics'], index['index'], marker='o', linestyle='-', color='red',
                                label='index (Line Chart)')
                ax2.set_ylabel('index', color='red')
                # #
                # # # Add data labels for the points in the line chart (df2)
                for x, y in zip(index['Demographics'], index['index']):
                    label = f'{y:.2f}'  # Format the label as desired
                    ax2.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='red',
                                 rotation=90)
                #
                # # Combine legends from both axes
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                # ax1.legend(lines + lines2, labels + labels2, loc='upper right')
                #
                # # plt.xticks(rotation=45)
                plt.tight_layout()
                #
                st.pyplot(plt)
                # # st.write("Top Ranking Segments:", top_segments_by_category)
                filtered_dict = {key: [value for value in values if value in selected_segments] for key, values in result_dict.items()}
                filtered_data = {key: value for key, value in filtered_dict.items() if (value and (isinstance(value, list) and len(value) > 0))}
                for key, values in filtered_data.items():
                    df = pd.DataFrame({key: values})
                    st.write(df)

        test=test.reset_index()
        test.columns=['Segments','Percentage']
        test = test.sort_values(by='Percentage', ascending=False)
        # Create a bar chart
        plt.figure(figsize=(15, 9))
        bars = plt.bar(test['Segments'], test['Percentage'])
        plt.xlabel('Segments')
        plt.ylabel('Percentage')
        plt.title('Segments-Profiling')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

        # Add data values on top of the bars
        for bar, value in zip(bars, test['Percentage']):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom',rotation=90)

        plt.tight_layout()  # Ensure all labels fit in the plot
        #
        st.pyplot(plt)
        # st.write(test)




if __name__=='__main__':
    main()
