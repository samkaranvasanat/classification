import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

st.set_page_config(
    page_title="Mushroom Classification App",
    page_icon="🍄",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f0f4f8;
    }
    .stButton>button {
        background-color: #b8e6d1;
        color: #2c3e50;
        border-radius: 10px;
        border: 2px solid #8fcbb3;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #8fcbb3;
        border-color: #b8e6d1;
        transform: scale(1.05);
    }
    .metric-card {
        background-color: #ffd7e4;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .title-text {
        color: #4a4a4a;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(45deg, #b8e6d1, #ffd7e4);
        border-radius: 15px;
        margin-bottom: 30px;
    }
    .subtitle-text {
        color: #6a6a6a;
        font-size: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .stSelectbox, .sidebar .stSlider {
        background-color: #e8f4ea;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    ################ Step 1 Create Web Title #####################
    # Title with custom styling
    st.markdown('<div class="title-text">🍄 Mushroom Classification App 🍄</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Is this mushroom edible or poisonous?</div>', unsafe_allow_html=True)

    ############### Step 2 Load dataset and Preprocessing data ##########

    @st.cache_data(persist=True)
    def load_data():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        file_path = os.path.join(DATA_DIR, 'mushrooms.csv')

        data = pd.read_csv(file_path)
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def spliting_data(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, display_labels=class_names)
            st.pyplot(fig)
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
      
    # df = load_data()
    # Add a fun loading animation
    with st.spinner('🍄 Loading magical mushroom data... 🌟'):
        df = load_data()

    # Success message after loading
    st.success('Data loaded successfully! Let\'s start classifying mushrooms! 🎉')

    # Create columns for layout
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("""
        ### 🎯 How to use:
        1. Choose a classifier
        2. Adjust parameters
        3. Select metrics
        4. Click classify!
        """)
    x_train, x_test, y_train, y_test = spliting_data(df)
    class_names = ['edible','poisonous']
    st.sidebar.markdown("""
    <div style='background-color: #ffd7e4; padding: 20px; border-radius: 10px;'>
        <h2 style='color: #4a4a4a; text-align: center;'>🎮 Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)

    classifier = st.sidebar.selectbox(
        "🤖 Choose Your Classifier",
        ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"),
        help="Select the machine learning algorithm you want to use"
    )

    # Add fun facts about mushrooms
    with st.expander("🍄 Fun Facts About Mushrooms"):
        st.markdown("""
        * The largest living organism on Earth is a honey fungus measuring 2.4 miles (3.8 km) across!
        * Mushrooms are more closely related to humans than to plants
        * Some mushrooms glow in the dark
        * There are over 14,000 species of mushrooms
        """)

    ############### Step 3 Train a SVM Classifier ##########

    # Modified classifier sections with enhanced styling
    if classifier == 'Support Vector Machine (SVM)':
        with st.sidebar.container():
            st.markdown('<div style="background-color: #e8f4ea; padding: 20px; border-radius: 10px;">', unsafe_allow_html=True)
            st.subheader("🎛️ Model Hyperparameters")
            C = st.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
            kernel = st.radio("🔧 Kernel", ("rbf", "linear"), key='kernel')
            gamma = st.radio("📊 Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
            st.markdown('</div>', unsafe_allow_html=True)

        metrics = st.sidebar.multiselect("📈 Performance Metrics", 
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("🚀 Classify!", key='classify'):
            with st.spinner('Training your model... 🤖'):
                model = SVC(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                
                precision = precision_score(y_test, y_pred).round(2)
                recall = recall_score(y_test, y_pred).round(2)
                
                st.write("Accuracy: ", round(accuracy, 2))
                st.write("Precision: ", precision)
                st.write("Recall: ", recall)
                plot_metrics(metrics)

    ############### Step 4 Training a Logistic Regression Classifier ##########

    if classifier == 'Logistic Regression':
        with st.sidebar.container():
            st.markdown('<div style="background-color: #e8f4ea; padding: 20px; border-radius: 10px;">', unsafe_allow_html=True)
            st.subheader("🎛️ Model Hyperparameters")
            C = st.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
            max_iter = st.slider("Maximum number of iterations", 100, 500, key='max_iter')
            gamma = st.radio("📊 Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
            st.markdown('</div>', unsafe_allow_html=True)

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("🚀 Classify!", key='classify'):
            with st.spinner('Training your model... 🤖'):
                st.subheader("Logistic Regression Results")
                model = LogisticRegression(C=C, max_iter=max_iter)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                
                precision = precision_score(y_test, y_pred).round(2)
                recall = recall_score(y_test, y_pred).round(2)
                
                st.write("Accuracy: ", round(accuracy, 2))
                st.write("Precision: ", precision)
                st.write("Recall: ", recall)
                plot_metrics(metrics)

    ############### Step 5 Training a Random Forest Classifier ##########

    if classifier == 'Random Forest':
        with st.sidebar.container():
            st.markdown('<div style="background-color: #e8f4ea; padding: 20px; border-radius: 10px;">', unsafe_allow_html=True)
            st.subheader("🎛️ Model Hyperparameters")
            n_estimators = st.number_input("Number of trees", 100, 5000, step=10, key='n_estimators')
            max_depth = st.number_input("Maximum depth", 1, 20, step=1, key='max_depth')
            bootstrap = st.checkbox(" 🔧 Bootstrap samples when building trees", True, key='bootstrap')
            st.markdown('</div>', unsafe_allow_html=True)
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("🚀 Classify!", key='classify'):
            with st.spinner('Training your model... 🤖'):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                
                precision = precision_score(y_test, y_pred).round(2)
                recall = recall_score(y_test, y_pred).round(2)
                
                st.write("Accuracy: ", round(accuracy, 2))
                st.write("Precision: ", precision)
                st.write("Recall: ", recall)
                plot_metrics(metrics)

     # Add a feature importance plot for Random Forest
    if classifier == 'Random Forest' and 'model' in locals():
        st.subheader("🌟 Feature Importance")
        feature_importance = pd.DataFrame(
            model.feature_importances_,
            index=x_train.columns,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance.head(10).plot(kind='bar', ax=ax)
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        st.pyplot(fig)

    # Add download button for predictions
    if 'y_pred' in locals():
        predictions_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        st.download_button(
            label="📥 Download Predictions",
            data=predictions_df.to_csv().encode('utf-8'),
            file_name='mushroom_predictions.csv',
            mime='text/csv'
        )

    # Modified data viewer
    if st.sidebar.checkbox("📊 Show Dataset", False):
        st.subheader("🍄 Mushroom Dataset Explorer")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Dataset Shape:", df.shape)
        with col2:
            st.write("Number of Classes:", len(df['type'].unique()))
        
        st.dataframe(df)

    # if st.sidebar.checkbox("Show raw data", False):
    #     st.subheader("Mushroom dataset")
    #     st.write(df)

if __name__ == '__main__':
    main()