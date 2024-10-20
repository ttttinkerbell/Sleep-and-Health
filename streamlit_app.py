import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import io
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

page = st.sidebar.selectbox("Select Page:", ["Homepage", "Introduction", "IDA", "EDA", "Conclusion"])

if page == "Homepage":
    st.title("Sleep Quality and Health 😴💤")
    st.write(
        """ 
            
        """
    )
    st.image("sleep.jpg", use_column_width=True)

elif page == "Introduction":
    st.title("Introduction 😄")
    st.markdown("# Hello!")
    st.write(
        """ 
            This is a project that analyzes the relationship between factors affecting sleep quality and health.
        """
    )
    st.write(
        """ 
            Through this analysis, we want to provide relevant healthcare practitioners with evidence on which influencing factors are important. 
        """
    )
    st.write(
        """
            At the end, we will also provide some conclusions that are different from common sense to prove that some "popularly believed" factors are actually not important. 
        """
    )
    st.write(
        """
            Hope this is helpful to medical practitioners. Now, let's get started.
        """
    )
    st.image("sleep1.jpg", use_column_width=True)

elif page == "IDA":
    st.title("IDA 📊")
    DATA_URL_1 = ('Sleep-and-Health/dataset1_with_missing_values.csv')
    DATA_URL_2 = ('Sleep-and-Health/dataset2_with_missing_values.csv')
    df1 = pd.read_csv(DATA_URL_1)
    df = pd.read_csv(DATA_URL_2)
    df2 = pd.read_csv(DATA_URL_2)
    df2['Sleep Disorder'] = df2['Sleep Disorder'].fillna('No Disorder')

    df1_drop = df1.copy().drop(columns=['Gender','Bedtime','Wake-up Time'])
    df2_drop = df2.copy().drop(columns=['Gender','Occupation'])

    def detect_outliers_iqr(df):
        outliers_dict = {}
    
        # Loop through each column
        for col in df.select_dtypes(include=['float64', 'int64']):
            # Calculate Q1, Q3, and IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
        
            # Define outliers based on the IQR rule
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        
            # Store the number of outliers and details in the dictionary
            outliers_dict[col] = {'Number of Outliers': len(outliers), 'Outliers': outliers}
    
        return outliers_dict

    outliers_info_1 = detect_outliers_iqr(df1_drop)
    outliers_info_2 = detect_outliers_iqr(df2_drop)

    def handle_mcar_missing_values(df):
        df_imputed = df.copy()
    
        # Numerical columns: fill missing with median
        numerical_columns = df_imputed.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
    
        # Categorical columns: fill missing with most frequent value
        categorical_columns = df_imputed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
    
        return df_imputed

    # Filling the missing values of df2 by KNN imputation
    def handle_mnar_missing_values(df):
        df_imputed = df.copy()

        # Encode categorical variables to numeric for KNN Imputation
        le = LabelEncoder()
        for col in df_imputed.select_dtypes(include=['object']).columns:
            df_imputed[col] = le.fit_transform(df_imputed[col].astype(str))
        
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(df_imputed), columns=df_imputed.columns)
    
        return df_imputed

    # Apply imputation on the datasets
    df1_filled = handle_mcar_missing_values(df1_drop)
    df2_filled = handle_mnar_missing_values(df2_drop)

    # Encode the first dataset. We don't have to do encoding for the second dataset since we already did the label encoding during KNN imputation.
    
    # One hot encoding for two columns: Sleep Disorders and Medication Usage.
    columns_to_encode = ['Sleep Disorders', 'Medication Usage']  
    df1_encoding = pd.get_dummies(df1_filled, columns=columns_to_encode, drop_first=True, dtype=float)
    
    # Then, do label encoding for another two columns: Physical Activity Level and Dietary Habits
    label_encoder = LabelEncoder()
    columns_to_encode2 = ['Physical Activity Level', 'Dietary Habits']
    # Apply label encoding to each column in the list
    for col in columns_to_encode2:
        df1_encoding[col] = label_encoder.fit_transform(df1_encoding[col])

    df2_encoding = df2_filled.copy()

    # Merge
    df1_encoding_copy = df1_encoding.copy()
    df2_encoding_copy = df2_encoding.copy()
    df1_unique_steps = df1_encoding_copy.drop_duplicates(subset='Daily Steps')
    df2_encoding_copy['calories_burned_daily'] = df2_encoding_copy['Daily Steps'].map(
        df1_unique_steps.set_index('Daily Steps')['Calories Burned']
    )
    # Final new data set
    df = df2_encoding_copy.copy()

    # Remove outliers
    Q1 = df['Heart Rate'].quantile(0.25)
    Q3 = df['Heart Rate'].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[(df['Heart Rate'] < (Q1 - 1.5 * IQR)) | (df['Heart Rate'] > (Q3 + 1.5 * IQR))]

    df = df[((df['Heart Rate'] >= (Q1 - 1.5 * IQR)) & 
                          (df['Heart Rate'] <= (Q3 + 1.5 * IQR)))]

    df_new = df.copy()
    df_new = handle_mnar_missing_values(df_new)

    page = st.sidebar.selectbox("Select subpage:", ["Load Dataset", "Precheck", "Data Processing"])

    if page == "Load Dataset":
        st.write("Let's load our two datasets first!")

        df1
        df

        st.write(
            """
            """
        )
        st.write("""For the NaN value in column "Sleep Disorder" of the second dataset, it means that the person doesn't have a sleep disorder.""")
        st.write("""Replace the NaN value with the character "No Disorder".""")

        df2

    elif page == "Precheck":
        st.write("Let's do some initial data analysis before we proceed with further analysis.")

        st.write("# Useless columns")
        st.write("Drop the useless columns.")
        df1_drop
        df2_drop

        st.write("# Missing value")
        st.write("Check if there are missing value in the datasets.")
        
        fig1, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(df1_drop.isnull(), cbar=False, cmap='viridis', ax=ax)
        ax.set_title("Heatmap of Missing Data in Dataset 1")
        st.pyplot(fig1)
        st.write(df1_drop.describe())

        fig2, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(df2_drop.isnull(), cbar=False, cmap='viridis', ax=ax)
        ax.set_title("Heatmap of Missing Data in Dataset 2")
        st.pyplot(fig2)
        st.write(df2_drop.describe())

        st.write("Check the correlation of missingness with other variables.")
        missing_corr_df1 = df1_drop.isnull().corr()
        missing_corr_df2 = df2_drop.isnull().corr()

        fig1, ax1 = plt.subplots(figsize=(9, 4))
        sns.heatmap(missing_corr_df1, annot=True, cmap='coolwarm', ax=ax1)
        ax1.set_title('Missing Data Correlation for Dataset 1')

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        sns.heatmap(missing_corr_df2, annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title('Missing Data Correlation for Dataset 2')

        st.pyplot(fig1)
        st.pyplot(fig2)

        st.write("Missing types for both datasets")
        st.write("First dataset: Missing Completely at Random (MCAR)")
        st.write("Second dataset: Missing Not at Random (MNAR)")

        st.write("# Duplicates")
        st.write("Number of duplicates in dataset 1:")
        duplicates1 = df1_drop.duplicated().sum()
        duplicates1
        st.write("Number of duplicates in dataset 2:")
        duplicates2 = df2_drop.duplicated().sum()
        duplicates2

        st.write("# Data Type")
        st.write("Check the data type of each columns in both datasets")
        buffer1 = io.StringIO()
        df1_drop.info(buf=buffer1)
        info_1 = buffer1.getvalue()
        st.text(info_1)

        buffer2 = io.StringIO()
        df2_drop.info(buf=buffer2)
        info_2 = buffer2.getvalue()
        st.text(info_2)

        st.write("# Outlier")
        st.write("Outlier detection using IQR method")

        # Create a selection box with column names
        selected_column_1 = st.selectbox("Select a column to check dataset 1 outliers", options=list(outliers_info_1.keys()))
        selected_column_2 = st.selectbox("Select a column to check dataset 2 outliers", options=list(outliers_info_2.keys()))

        if selected_column_1:
            st.subheader(f"Outlier details for column: {selected_column_1}")
            st.write(f"Number of Outliers: {outliers_info_1[selected_column_1]['Number of Outliers']}")
    
            if outliers_info_1[selected_column_1]['Number of Outliers'] > 0:
                st.write("Outlier Rows:")
                st.dataframe(outliers_info_1[selected_column_1]['Outliers'])
            else:
                st.write("No outliers detected in this column.")

        if selected_column_2:
            st.subheader(f"Outlier details for column: {selected_column_2}")
            st.write(f"Number of Outliers: {outliers_info_2[selected_column_2]['Number of Outliers']}")
    
            if outliers_info_2[selected_column_2]['Number of Outliers'] > 0:
                st.write("Outlier Rows:")
                st.dataframe(outliers_info_2[selected_column_2]['Outliers'])
            else:
                st.write("No outliers detected in this column.")

        st.write(
            """
            """)
        st.write("We only find two columns with outliers.")
        st.write("For heart rate, we remove outliers. For sleep disorder, this is not abnormal because most people do not have sleep disorders.")  

    elif page == "Data Processing":
        st.write("Now, we need to do the follow-up processing.")

        st.write("# Handling missing values")
        st.write("We filling the missing values of dataset 1 by median and most frequent value")
        st.write("Here is the filled dataset：")
        df1_filled
        st.write(df1_filled.describe())

        st.write("Then we filling the missing values of dataset 2 by KNN imputation")
        st.write("Here is the filled dataset：")
        df2_filled
        st.write(df2_filled.describe())

        st.write("# Encoding")
        st.write("Here is the datasets after encoding:")
        df1_encoding
        df2_encoding

        st.write("# Marge")
        st.write("Unfortunately, my two data sets came from two completely different places and there is no correlation between them.")
        st.write("So I can't merge them directly.")
        st.write("But I tried to find some other way to merge some of the columns.")
        st.write("I only focus on some specific columns that have strong correlation, especially numeric columns.")
        st.write("So I extracted the required columns and added them to another data set.")
        # Correlation of two datasets
        df1_numeric = df1_encoding.select_dtypes(include=[float, int])
        df2_numeric = df2_encoding.select_dtypes(include=[float, int])

        corr_df1 = df1_numeric.corr()
        corr_df2 = df2_numeric.corr()

        # Heatmaps to visualize the correlation
        fig3, ax1 = plt.subplots(figsize=(9, 4))
        sns.heatmap(corr_df1, annot=True, cmap='coolwarm', ax=ax1)
        ax1.set_title('Correlation for Dataset 1')

        fig4, ax2 = plt.subplots(figsize=(9, 4))
        sns.heatmap(corr_df2, annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title('Correlation for Dataset 2')

        st.pyplot(fig3)
        st.pyplot(fig4)
        st.write("Idea of merging: ")
        st.write(
            """
            I'm interested in the impact of daily calories burned on sleep. 
            But this column is not included in the second dataset. 
            I noticed that calories burned and daily steps have an extremely high correlation, and that daily steps are included in both datasets. 
            So I plan to add calorie burn to the second dataset based on daily steps. 
            For the same value of daily steps, I will add the value of calories burned to the second data set."""
        )
        df

        st.write("# Remove outlier")
        buffer3 = io.StringIO()
        df.info(buf=buffer3)
        info_3 = buffer3.getvalue()
        st.text(info_3)
        st.write("""For missing data in column "calories_burned_daily", we imputed again.""")

        st.write("# Final dataset")
        st.write("Now, we have a preprocessed data set!")
        df_new
        st.write(df_new.describe())

elif page == "EDA":
    st.title("EDA 📈")
    st.write("Now, let's do some exploratory data analysis.")
    DATA_URL_1 = ('/workspaces/Sleep-and-Health/dataset1_with_missing_values.csv')
    DATA_URL_2 = ('/workspaces/Sleep-and-Health/dataset2_with_missing_values.csv')
    df1 = pd.read_csv(DATA_URL_1)
    df = pd.read_csv(DATA_URL_2)
    df2 = pd.read_csv(DATA_URL_2)
    df2['Sleep Disorder'] = df2['Sleep Disorder'].fillna('No Disorder')

    df1_drop = df1.copy().drop(columns=['Gender','Bedtime','Wake-up Time'])
    df2_drop = df2.copy().drop(columns=['Gender','Occupation'])

    def detect_outliers_iqr(df):
        outliers_dict = {}
    
        # Loop through each column
        for col in df.select_dtypes(include=['float64', 'int64']):
            # Calculate Q1, Q3, and IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
        
            # Define outliers based on the IQR rule
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        
            # Store the number of outliers and details in the dictionary
            outliers_dict[col] = {'Number of Outliers': len(outliers), 'Outliers': outliers}
    
        return outliers_dict

    outliers_info_1 = detect_outliers_iqr(df1_drop)
    outliers_info_2 = detect_outliers_iqr(df2_drop)

    def handle_mcar_missing_values(df):
        df_imputed = df.copy()
    
        # Numerical columns: fill missing with median
        numerical_columns = df_imputed.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
    
        # Categorical columns: fill missing with most frequent value
        categorical_columns = df_imputed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
    
        return df_imputed

    # Filling the missing values of df2 by KNN imputation
    def handle_mnar_missing_values(df):
        df_imputed = df.copy()

        # Encode categorical variables to numeric for KNN Imputation
        le = LabelEncoder()
        for col in df_imputed.select_dtypes(include=['object']).columns:
            df_imputed[col] = le.fit_transform(df_imputed[col].astype(str))
        
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(df_imputed), columns=df_imputed.columns)
    
        return df_imputed

    # Apply imputation on the datasets
    df1_filled = handle_mcar_missing_values(df1_drop)
    df2_filled = handle_mnar_missing_values(df2_drop)

    # Encode the first dataset. We don't have to do encoding for the second dataset since we already did the label encoding during KNN imputation.
    
    # One hot encoding for two columns: Sleep Disorders and Medication Usage.
    columns_to_encode = ['Sleep Disorders', 'Medication Usage']  
    df1_encoding = pd.get_dummies(df1_filled, columns=columns_to_encode, drop_first=True, dtype=float)
    
    # Then, do label encoding for another two columns: Physical Activity Level and Dietary Habits
    label_encoder = LabelEncoder()
    columns_to_encode2 = ['Physical Activity Level', 'Dietary Habits']
    # Apply label encoding to each column in the list
    for col in columns_to_encode2:
        df1_encoding[col] = label_encoder.fit_transform(df1_encoding[col])

    df2_encoding = df2_filled.copy()

    # Merge
    df1_encoding_copy = df1_encoding.copy()
    df2_encoding_copy = df2_encoding.copy()
    df1_unique_steps = df1_encoding_copy.drop_duplicates(subset='Daily Steps')
    df2_encoding_copy['calories_burned_daily'] = df2_encoding_copy['Daily Steps'].map(
        df1_unique_steps.set_index('Daily Steps')['Calories Burned']
    )
    # Final new data set
    df = df2_encoding_copy.copy()

    # Remove outliers
    Q1 = df['Heart Rate'].quantile(0.25)
    Q3 = df['Heart Rate'].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[(df['Heart Rate'] < (Q1 - 1.5 * IQR)) | (df['Heart Rate'] > (Q3 + 1.5 * IQR))]

    df = df[((df['Heart Rate'] >= (Q1 - 1.5 * IQR)) & 
                          (df['Heart Rate'] <= (Q3 + 1.5 * IQR)))]

    df_new = df.copy()
    df_new = handle_mnar_missing_values(df_new)

    page = st.sidebar.selectbox("Select subpage:", ["Histograms","Box plots","Scatter plots","Correlation heatmap","Linear Regression"])

    if page == "Histograms":
        st.write("# Histograms")
        # Get numeric columns
        numeric_columns = df_new.select_dtypes(include=['float64', 'int64']).columns
        selected_column = st.selectbox("Select a numeric column for histogram with KDE", numeric_columns)

        if selected_column:
            st.subheader(f"Histogram with KDE for {selected_column}")

            fig, ax = plt.subplots()
            sns.histplot(df[selected_column], kde=True, ax=ax, stat="density")
            ax.set_title(f"Histogram with KDE for {selected_column}")
            st.pyplot(fig)

        st.write("# Analysis")
        st.write("Except stress leve, all other variables show multimodal distribution.")
        st.write("This suggests that our data contain multiple subgroups or distinct patterns.")

    elif page == "Box plots":
        st.write("# Box plots")
        # Get numeric columns
        numeric_columns = df_new.select_dtypes(include=['float64', 'int64']).columns
        # Selection boxes for x-axis and y-axis columns
        x_column = st.selectbox("Select column for x-axis", numeric_columns)
        y_column = st.selectbox("Select column for y-axis", numeric_columns)
        
        if x_column and y_column:
            st.subheader(f"Box Plot for {x_column} vs {y_column}")

            # box Plot
            fig_box = px.box(df_new, x=x_column, y=y_column, title=f"Box Plot of {x_column} vs {y_column}")
            st.plotly_chart(fig_box)

        st.write("# Analysis")
        st.write("Most of the graphs show a scattering trend.")
        st.write("Only a few graphs, such as quality of sleep vs stress level, or sleep duration, show a more meaningful distribution.")
        

    elif page == "Scatter plots":
        st.write("# Scatter plots")
        # Get numeric columns
        numeric_columns = df_new.select_dtypes(include=['float64', 'int64']).columns
        # Selection boxes for x-axis and y-axis columns
        x_column = st.selectbox("Select column for x-axis", numeric_columns)
        y_column = st.selectbox("Select column for y-axis", numeric_columns)
        
        if x_column and y_column:
            st.subheader(f"Scatter Plot: {x_column} vs {y_column}")

            # Scatter Plot
            fig_scatter = px.scatter(df_new, x=x_column, y=y_column, title=f"Scatter Plot of {x_column} vs {y_column}")
            st.plotly_chart(fig_scatter)
        
        st.write("# Analysis")
        st.write("Similar information as box plot.")

    elif page == "Correlation heatmap":
        st.write("# Correlation matrix and heatmap")
        numeric_columns = df_new.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = df_new[numeric_columns].corr()
        st.write("Correlation matrix:")
        correlation_matrix

        # Visualizing the correlation matrix
        fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='Viridis'
            ))
        fig_corr.update_layout(title='Correlation Matrix Heatmap')
        st.plotly_chart(fig_corr)

        st.write("# Analysis")
        st.write("Since I study the factors that affect sleep, I looked at correlations between quality of sleep and other variables. ")
        st.write("The factor I found that has the biggest negative impact on sleep quality is stress level, followed by heart rate.")
        st.write("Sleep duration havs the biggest positive impact on sleep quality.")
        st.write("The least impact on sleep quality is calories burned daily, which is from daily step.")
        st.write("The second less impact on sleep quality is blood pressure.")
        st.write("""This result is different from the "common sense" we usually know. """)
        st.write("Because we generally say that people will feel tired or even sleepy after a day of high-intensity exercise.")

    elif page == "Linear Regression":
        st.write("# Linear Regression")
        st.write("Let's take the most obvious relationship as an example, i.e. Stress Level and Quality of Sleep.")
        # Split the data into features and target
        X = df_new[['Stress Level']]
        y = df_new['Quality of Sleep']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.write("### Regression Coefficients")
        st.write(f"Intercept: {model.intercept_}")
        st.write(f"Coefficient: {model.coef_[0]}")

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Evaluation")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R-squared (R²): {r2}")

        st.write("### Scatter Plot with Regression Line")
        fig, ax = plt.subplots()

        # Plot the training data
        sns.scatterplot(x=X_train['Stress Level'], y=y_train, ax=ax, label="Training Data", color="blue")

        # Plot the test data
        sns.scatterplot(x=X_test['Stress Level'], y=y_test, ax=ax, label="Test Data", color="green")

        # Plot the regression line
        sns.lineplot(x=X_test['Stress Level'], y=y_pred, ax=ax, color="red", label="Regression Line")

        # Show the plot
        ax.set_title("Linear Regression Plot")
        st.pyplot(fig)

        st.write("### Predictions vs Actual")
        pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.write(pred_df.head())


elif page == "Conclusion":
    st.title("Conclusion 💡")

    st.write(
        """
        This data set shows us several factors that have a strong impact on sleep.
        The factor that has the biggest negative impact on our sleep is stress levels.
        This reminds us to pay attention to releasing stress in daily life, otherwise it may cause people to have sleep disorders.
        Over time, this can greatly impact our health.
        The second thing to pay attention to is heart rate.
        This is especially likely to occur in older people, who are more likely to have heart problems.
        For people in the related medical field, we may be able to develop some medicine based on heart rate to improve the sleep quality of old people.
        """)

    st.write(
        """
        A factor that positively affects the quality of our sleep is sleep duration. 
        Apparently, the longer people sleep, the easier it is for them to avoid the health problems that come with sleep deprivation.
        If people suffer from poor sleep quality because they feel too stressed,
        they can compensate for this by increasing the sleep duration.
        """)

    st.write(
        """
        The factor that has the least impact on our sleep is calorie burned daily.
        This indirectly suggests that exercise does not necessarily improve people's sleep quality, even if people feel tired after doing exercise.
        """)
    st.image("sleep2.jpg", use_column_width=True)
    
