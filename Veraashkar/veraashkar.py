#I started with importing the needed libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime as dt
import  plotly.express as px
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
import calendar
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
matplotlib.use('Agg')
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False) #to disable errors in charts

#Design sidebar contents
st.sidebar.title("Supermarket Sales Data Analysis")
st.sidebar.markdown("An easy way to analyse supermarket sales data of three branches for 3 months")


def main():

    menu = ['Explore your dataset','Create some visuals','RFM Analysis','Machine Learning Algorithm','Upload your dataset']
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=='Explore your dataset':
        image= Image.open('supermarket.png')
        st.image(image,use_column_width=True)
        st.info("This dashboard is made to let you analyse your supermarket data and generate insights about sales, customer behaviour, payment types and many other options.")
        st.header("Explore your dataset")
        df=pd.read_csv("supermarketsales.csv",encoding="latin1")


        if st.checkbox("Show Dataset"):
            number=st.number_input("Number of Rows to view",5,15)
            st.dataframe(df.head(number))
            st.success("Data loaded successfully")


            data_dim= st.radio("Shape of the dataset:", ("Number of Rows","Number of Columns"))
            if st.button("Show Dimension"):
                if data_dim== 'Number of Rows':
                    st.write(df.shape[0])
                elif data_dim== 'Number of Columns':
                    st.write(df.shape[1])
                else:
                    st.write(df.shape)

            Info =['Dataset Information','Display Multiple Columns','Display the dataset summary','Check for missing values in the dataset','Display Customer distribution by City']
            options=st.selectbox("Know More",Info)


            if options==('Dataset Information'):
                st.markdown("**InvoiceID**: Computer-generated nvoice identification number.")
                st.markdown("**BranchF**:Branches of the Supermarke)t.")
                st.markdown("**City**: Supermarkets location.")
                st.markdown("**Customer Type**: Members are member card holders and normal without member cards.")
                st.markdown("**Gender**: Gender type of customer.")
                st.markdown("**Product line**: General item categorization groups.")
                st.markdown("**UnitPrice**: Price of each product in U.S. Dollars.")
                st.markdown("**Quantity**: Number of products purchased by customer.")
                st.markdown("**Tax**: 5% tax fee on total amount.")
                st.markdown("**TotalSales**: Total price including tax.")
                st.markdown("**Date**: Date of purchase.")
                st.markdown("**Time**: Purchase time.")
                st.markdown("**Payment**: Payment method used by customer for purchase.")
                st.markdown("**COGS**: Cost of goods sold.")
                st.markdown("**Gross margin percentage**: Gross margin percentage.")
                st.markdown("**Gross income**: Gross income.")
                st.markdown("**Rating**:Customer stratification rating for shopping experience(On a scale of 1 to 10).")


            if options=='Display Multiple Columns':
                 selected_columns=st.multiselect('Select Preferred Columns:',df.columns)
                 df1=df[selected_columns]
                 st.dataframe(df1)

            if options=='Check for missing values in the dataset':
                 st.write(df.isnull().sum(axis=0))
                 if st.button("Drop Null Values"):
                     df=df.dropna()
                     st.success("Null values droped successfully")


            if options=='Display the dataset summary':
                 st.write(df.describe().T)

            if options=='Insights obtained from the summary table':
                 d=df.describe().T
                 st.write("1)** Average Unit Price** of an article in the supermaket is $",round(d.iloc[0,1],2),".")
                 st.write("2)** Average Gross Margin Percentage** of articles sold is %",round(d.iloc[5,1],2),".")
                 st.write("3)** Average Gross Income** of articles sold in the supermarket is $",round(d.iloc[6,1],2),".")
                 st.write("4)** Average rating** of an experience in the supermaket is",round(d.iloc[-1,1],2),".")

            if options=='Display Customer distribution by City':
                 city_cust=df[['City','CustomerID']].drop_duplicates()
                 city_cust =city_cust.groupby(['City'])['CustomerID'].aggregate('count').reset_index()
                 city_invoice=df[['City','InvoiceID']].drop_duplicates()
                 city_invoice =city_invoice.groupby(['City'])['InvoiceID'].aggregate('count').reset_index()
                 data =pd.merge(city_cust, city_invoice, on='City', how='inner')
                 st.dataframe(data)
                 st.info("Mandalay City has a large number of unique customer visits. Yangon City has a larger number of invoices however fewer customers coming in. Which means that **Yangon City** had better customer retention than **Mandalay City**.")


    elif choice=='Create some visuals':
        image= Image.open('visuals.jpeg')
        st.image(image,use_column_width=True)
        st.header("Create some visuals")
        df=pd.read_csv("supermarketsales.csv",encoding="latin1")
            #st.dataframe(df.head(50))
        df['Date']=pd.to_datetime(df['Date'])

        df['Month']=pd.DatetimeIndex(df['Date']).month

        df['MonthName'] = df['Month'].apply(lambda x: calendar.month_abbr[x])
        df['Time'] = pd.to_datetime(df['Time'])
        df['Hour'] = (df['Time']).dt.hour


        if st.button("Show Dataset again"):
            st.dataframe(df.head(50))

        col1,col2,col3=st.beta_columns(3)

        st.subheader("Customizable Plots")
        with col1:
            measure_selection = st.selectbox('Choose a Measure:', ['Quantity','Unitprice','TotalSales','cogs','gross_income'], key='1')
        with col2:
            fact_selection = st.selectbox('Choose a Fact:', ['Productline','City','Payment','Gender','Customertype','Branch','MonthName','Hour'], key='1')
            ax=df.groupby([fact_selection])[measure_selection].aggregate('sum').reset_index().sort_values(measure_selection,ascending=False)
            cust_data=ax
        with col3:
            type_of_plot=st.selectbox("Select Type of Plot",["Bar Chart","Horizontal Bar"])

        col4,col5= st.beta_columns((1,2))
        #col4,col5=st.beta_columns(2)
        with col4:
            if st.button("Generate Plot"):
                if type_of_plot=='Bar Chart':
                    st.success("Generating Customizable Plot of {} Type for {} relative to {}".format(type_of_plot,measure_selection,fact_selection))

                    plt.xticks(rotation=45)
                    plt.autoscale()
                    plt.tight_layout(rect=(0, 0.25, 1, 1))
                    plt.bar(ax[fact_selection], ax[measure_selection], align='center')
                    plt.ylabel(measure_selection)
                    st.pyplot()


                elif type_of_plot=='Horizontal Bar':
                    st.success("Generating Customizable Plot of {} Type for {} relative to {}".format(type_of_plot,measure_selection,fact_selection))
                    plt.barh(ax[fact_selection], ax[measure_selection])
                    st.pyplot()



        with col5:
            if st.checkbox('Donut Chart'):
                st.success("Generating Customizable Plot of {} Type representing the distribution of {} by {}".format(type_of_plot,fact_selection,measure_selection))
                labels = ax[fact_selection].unique()
                values =df.groupby([fact_selection])[measure_selection].sum()
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
                st.plotly_chart(fig)


    elif choice=='RFM Analysis':
        image= Image.open('RFM.png')
        st.image(image,use_column_width=True)
        st.header("RFM Analysis")


        df=pd.read_csv("supermarketsales.csv",encoding="latin1")


        df['Date']=pd.to_datetime(df['Date'])
        #st.dataframe(df.head(50))
        #st.write(df['Date'].min(),df['Date'].max())


        #Convert the Obejct date field to datetime
        Latest_Date=dt.datetime(2019,3,30)

        #Calculating The Recency, Frequency,and Monetary Values
        RFMScores = df.groupby('CustomerID').agg({'Date': lambda x: (Latest_Date - x.max()).days, 'InvoiceID': lambda x: len(x), 'TotalSales': lambda x: x.sum()})

        #Converting the Order Date Col into integer
        RFMScores['Date'] = RFMScores['Date'].astype(int)

        #Rename column names to Recency, Frequency and Monetary
        RFMScores.rename(columns={'Date': 'Recency',
                    'InvoiceID': 'Frequency',
                    'TotalSales': 'Monetary'}, inplace=True)

        quantiles =RFMScores[['Recency', 'Frequency', 'Monetary']].quantile([.2, .4, .6, .8]).to_dict()
        #st.write(quantiles)

        def r_score(x):
            if x <= quantiles['Recency'][.2]:
                return 5
            elif x <= quantiles['Recency'][.4]:
                return 4
            elif x <= quantiles['Recency'][.6]:
                return 3
            elif x <= quantiles['Recency'][.8]:
                return 2
            else:
                return 1

        def fm_score(x, c):
            if x <= quantiles[c][.2]:
                return 1
            elif x <= quantiles[c][.4]:
                return 2
            elif x <= quantiles[c][.6]:
                return 3
            elif x <= quantiles[c][.8]:
                return 4
            else:
                return 5

        rfmseg = RFMScores
        rfmseg['R'] =rfmseg['Recency'].apply(lambda x: r_score(x))
        rfmseg['F'] =rfmseg['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
        rfmseg['M'] =rfmseg['Monetary'].apply(lambda x: fm_score(x, 'Monetary'))

        #rfmseg['RFM_Score'] = rfmseg.R.map(str) \
                                    #+ rfmseg.F.map(str) \
                                    #+ rfmseg.M.map(str)

        #RFMScores['RFM_Score'] =RFMScores['R'].astype(str)+ RFMScores['F'].astype(str)+ RFMScores['M'].astype(str)
        #st.dataframe(RFMScores.head())
        #st.write(RFMScores.dtypes)

        RFMScores['RFM_Score'] =RFMScores['R'] +RFMScores['F'] +RFMScores['M']


        #RFMScores['RFM_Score'] =RFMScores['R'].astype(str) +RFMScores['F'].astype(str)  +RFMScores['M'].astype(str)
        #Every time i try to write RFM scores without adding its value by converting them to str:
        #i get TypeError: unsupported operand type(s) for -: 'str' and 'str'


        st.markdown("The **RFM** (Recency, Frequency, Monetary) Analysis Marketing Model is used to identify a company's or an organization's best customers by using certain measures. The RFM model is based on three quantitative factors:")
        st.markdown("**Recency:** How recently a customer has made a purchase")
        st.markdown("**Frequency:** How frequently does the customer makes a purchase")
        st.markdown("**Monetary Value:** How much money has the customer spent")

        st.info("Let's take a look of our dataset after applying RFM Analysis")

        score_lables= ['Vulnerable','Needs Attention','Propitious','Potential Loyalist','Loyal']
        RFMScores["Segment"] =pd.qcut(RFMScores.RFM_Score,q=5,labels=score_lables).astype('object')            #st.write(score_groups.values)
        st.dataframe(RFMScores.head(10))

        if st.checkbox("Explain More"):
            st.write("**Loyal Customers** are customers who bought recently, buy most often and spend the most.")
            st.write("**Vulnerable Customers** are at risk, who have purchased a long time ago and are required to bring them back!")
            st.write("**Propitious Customers** are recent shoppers with above average spending.")
            st.write("**Customers needing attention** are characterized with average recency,frequency, and monertary values.")
            st.write("**Potential Loyalist Customers** are recent customers with above average frequency.")


        if st.checkbox("Plot RFM Result"):
            colors = ['cyan'] * 10
            colors[1] = 'yellow'

            fig = go.Figure(data=[go.Bar(
            x=RFMScores['Segment'].unique(),
            y=RFMScores['Segment'].value_counts(),
            marker_color=colors
            )])
            fig.update_layout(title_text='RFM Market Segmentation')
            st.plotly_chart(fig)

        if st.checkbox("Show result as a Percent"):
            s = RFMScores.Segment
            counts = s.value_counts()
            percent = s.value_counts(normalize=True)
            percent100 = s.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            s= pd.DataFrame({'counts': counts, 'per100': percent100})
            st.dataframe(s)
            Pot = s.iloc[0,1]
            Vun =s.iloc[1,1]
            Att =s.iloc[2,1]
            Pro=s.iloc[3,1]
            Loy=s.iloc[4,1]
            st.warning("Loyal customers percentage is the highest. However, 24.4% of customer are vulnerable customers, and this risk is going to get bigger if no immediate strategy is applied")


    elif choice=='Machine Learning Algorithm':

        df=pd.read_csv("supermarketsales.csv",encoding="latin1")


        df['Date']=pd.to_datetime(df['Date'])
        #st.dataframe(df.head(50))

        #st.write("The first order is",df['Date'].min(), "and the latest order date is", df['Date'].max())

        #Convert the Obejct date field to datetime
        Latest_Date=dt.datetime(2019,3,30)

        #Calculating The Recency, Frequency,&Monetary Values
        RFMScores = df.groupby('CustomerID').agg({'Date': lambda x: (Latest_Date - x.max()).days, 'InvoiceID': lambda x: len(x), 'TotalSales': lambda x: x.sum()})

        #Converting the Order Date Col into integer
        RFMScores['Date'] = RFMScores['Date'].astype(int)

        #Rename column names to Recency, Frequency and Monetary
        RFMScores.rename(columns={'Date': 'Recency',
                    'InvoiceID': 'Frequency',
                    'TotalSales': 'Monetary'}, inplace=True)

        quantiles =RFMScores[['Recency', 'Frequency', 'Monetary']].quantile([.2, .4, .6, .8]).to_dict()
        #st.write(quantiles)

        def r_score(x):
            if x <= quantiles['Recency'][.2]:
                return 5
            elif x <= quantiles['Recency'][.4]:
                return 4
            elif x <= quantiles['Recency'][.6]:
                return 3
            elif x <= quantiles['Recency'][.8]:
                return 2
            else:
                return 1

        def fm_score(x, c):
            if x <= quantiles[c][.2]:
                return 1
            elif x <= quantiles[c][.4]:
                return 2
            elif x <= quantiles[c][.6]:
                return 3
            elif x <= quantiles[c][.8]:
                return 4
            else:
                return 5

        RFMScores['R'] =RFMScores['Recency'].apply(lambda x: r_score(x))
        RFMScores['F'] =RFMScores['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
        RFMScores['M'] =RFMScores['Monetary'].apply(lambda x: fm_score(x, 'Monetary'))

        RFMScores['RFM_Score'] =(RFMScores['R'].astype(int) +RFMScores['F'].astype(int) +RFMScores['M'].astype(int))/3
        #st.dataframe(RFMScores.head())
        #st.write(RFMScores.dtypes)

        score_lables= ['Vulnerable','Needs Attention','Propitious','Potential Loyalist','Loyal']
        RFMScores["Segment"] =pd.qcut(RFMScores.RFM_Score,q=5,labels=score_lables).astype('object')            #st.write(score_groups.values)
        #RFMScores['RFM Level']=score_groups.values
        #st.dataframe(RFMScores.head())

        #RFMScores_Segment = RFMScores.groupby('Segment')
        #st.write(RFMScores_Segment.head())

        #levels = list(RFMScores_Segment.Segment)
        #score = list(RFMScores_Segment.counts)
        #plt.figure(figsize=(12,8))
        #plt.title('Customer Levels distribution')
        #squarify.plot(sizes=score, label=levels)
        #st.pyplot()



        #Modeling
        image= Image.open('KMeans.jpg')
        st.image(image)
        st.header("** Machine Learning algorithm: K-means Clustering**")
        st.subheader("What is K-means clustering and how does it help?")
        st.write("K-means clustering is an unsupervised machine learning that their purpose is to group observations that have similar characteristics, which means to group data points into distinct non-overlapping subgroups. We are using it to segment the supermarket customers to get a better understanding of them.")
        def neg_to_zero(x):
            if x <= 0:
                return 1
            else:
                return x

        RFMScores['Recency'] = [neg_to_zero(x) for x in RFMScores.Recency]

        rfm_log = RFMScores[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(rfm_log)
        rfm_normalized= scaler.transform(rfm_log)

        rfm_Scaled = pd.DataFrame(rfm_normalized, index = RFMScores.index, columns = rfm_log.columns)
        plt.figure(figsize=(10,8))

        st.info("We need to define a target number k, which refers to the point where the interia starts to have a stable decrease. Inertia is the sum of squared distances of samples to their closest cluster center")
        st.subheader("The Elbow curve is one of the most popular methods to determine this optimal value of k")


        kls = np.arange(1,11,1)
        inertias=[]
        for k in kls :
            knc = KMeans(n_clusters=k, random_state=50)
            knc.fit(rfm_Scaled)
            inertias.append(knc.inertia_)

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(kls, inertias,'--o', markersize=22, color='turquoise')
        plt.xlabel('Clusters')
        plt.ylabel('Inertia')
        plt.xticks(kls)
        st.pyplot()

        st.info("Here we can conclude that the best k value is 5")
        # Silhouette analysis
        st.subheader("**Selecting the number of clusters with silhouette analysis**")
        st.write("Silhouette analysis is used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters. The value of the silhouette score range lies between -1 to 1.")

        if st.checkbox("Let's check which number of cluster is the best for our data"):
            clusters = [2, 3, 4, 5, 6, 7, 8,9,10]
            for num_clusters in clusters:
                kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
                kmeans.fit(rfm_Scaled)
                cluster_labels = kmeans.labels_
                # silhouette score
                silhouette_avg = silhouette_score(rfm_Scaled, cluster_labels)
                st.write("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

            st.info("**Optimal Choice of Clusters is 5**")
            kmeans = KMeans(n_clusters=5, max_iter=50)
            kmeans.fit(rfm_Scaled)
            # assign the label
            RFMScores['Cluster_Id'] = kmeans.labels_

            st.dataframe(RFMScores.head(50))
            km = KMeans(n_clusters=5)
            clusters = km.fit_predict(RFMScores.iloc[:,0:3])
            RFMScores["label"] = clusters

            st.subheader("**3D Plot Visualization that shows the data points calssified into 5 classes**")
            fig = plt.figure(figsize=(20,10),)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(RFMScores.Recency[RFMScores.label == 0],RFMScores["Monetary"][RFMScores.label == 0],RFMScores["Frequency"][RFMScores.label == 0], c='cyan', s=100,label='0')
            ax.scatter(RFMScores.Recency[RFMScores.label == 1],RFMScores["Monetary"][RFMScores.label == 1],RFMScores["Frequency"][RFMScores.label == 1], c='turquoise', s=100,label='1')
            ax.scatter(RFMScores.Recency[RFMScores.label == 2],RFMScores["Monetary"][RFMScores.label == 2],RFMScores["Frequency"][RFMScores.label == 2], c='blue', s=100,label='2')
            ax.scatter(RFMScores.Recency[RFMScores.label == 3],RFMScores["Monetary"][RFMScores.label == 3],RFMScores["Frequency"][RFMScores.label == 3], c='yellow', s=100,label='3')
            ax.scatter(RFMScores.Recency[RFMScores.label == 4],RFMScores["Monetary"][RFMScores.label == 4],RFMScores["Frequency"][RFMScores.label == 4], c='gray', s=100,label='4')
            ax.view_init(30, 185)
            plt.xlabel("Recency")
            plt.ylabel("Monetary")
            ax.set_zlabel('Frequency')
            ax.legend(prop={'size':30})
            st.pyplot()



    elif choice=='Upload your dataset':
            image= Image.open('analysis.jpg')
            st.image(image,use_column_width=True)
            st.header("**Upload your dataset**")
            st.markdown("Upload your data here to get a similar analysis and dashboard as this one.")

            data=st.file_uploader("Browse to choose your file or drag an drop it here:",type=['csv','xlsx','txt','json'])
            if data is not None:
                df=pd.read_csv(data)
                st.dataframe(df.head(5))
                with st.spinner('Loading...'):
                    time.sleep(5)
                st.success('Great! We will get back to you once your dashboard is ready')



if __name__ == '__main__':
    main()
