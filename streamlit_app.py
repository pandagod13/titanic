import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    return df

# Preprocess data
def preprocess_data(df):
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df['Last_Name'] = df['Name'].apply(lambda x: x.split(',')[0].strip())
    df['Title'] = df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Jonkheer', 'Don', 'Sir', 'Capt', 'Lady', 'the Countess', 'Dona', 'Mlle', 'Mme'], 'Rare')
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Age_cat'] = df['Age'].apply(lambda x: 'Kid' if x <= 15 else ('Adult' if x <= 60 else 'Elderly'))
    df['Deck'] = df['Cabin'].apply(lambda x: 'Unknown' if pd.isnull(x) else x[0])
    df['Sex_bin'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Sex_Age_cat'] = df['Sex'] + '_' + df['Age_cat'].astype(str)
    df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Deck', 'Title', 'Age_cat'], drop_first=True)
    features_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Last_Name']
    df_encoded = df_encoded.drop(columns=features_to_drop)
    
    # Ensure all features are numeric and handle missing values
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return df, df_encoded

# Plot bar chart
def bar_chart(df, feature):
    survived = df[df['Survived'] == 1][feature].value_counts()
    dead = df[df['Survived'] == 0][feature].value_counts()
    df_bar = pd.DataFrame([survived, dead])
    df_bar.index = ['Survived', 'Dead']
    df_bar.plot(kind='bar', stacked=True, figsize=(10, 5))
    st.pyplot(plt)

# Train and evaluate model
def train_evaluate_model(df_encoded):
    X = df_encoded.drop(columns=['Survived'])
    y = df_encoded['Survived']
    
    # Ensure all features are numeric and handle missing values
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return model, accuracy, conf_matrix, class_report

# Visualize decision tree
def visualize_decision_tree(df_encoded):
    Bt_features = ['Pclass', 'Age', 'Sex_bin']
    # Check if all Bt_features are in df_encoded
    missing_features = [feature for feature in Bt_features if feature not in df_encoded.columns]
    if missing_features:
        st.error(f"Missing features in the dataset: {missing_features}")
        return
    X = df_encoded[Bt_features]
    y = df_encoded['Survived']
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X, y)
    plt.figure(figsize=(15, 10))
    plot_tree(dt, feature_names=Bt_features, class_names=['Not Survived', 'Survived'], filled=True, rounded=True, fontsize=12)
    st.pyplot(plt)

# Main function to set up the dashboard
def main():
    st.title("Titanic Survival Prediction Dashboard")
    
    df = load_data()
    df, df_encoded = preprocess_data(df)
    
    st.header("Data Overview")
    st.write(df.head())
    
    st.header("Bar Chart")
    feature = st.selectbox("Select feature to plot", ['Sex', 'Pclass', 'Age_cat', 'family_size', 'Deck'])
    bar_chart(df, feature)
    
    st.header("Survival Rate by Gender and Age Category")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sex_Age_cat', y='Survived', data=df)
    plt.title('Survival Rate by Gender and Age Category')
    plt.ylabel('Survival Rate')
    plt.xlabel('Sex_Age_cat')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    
    st.header("Model Training and Evaluation")
    model, accuracy, conf_matrix, class_report = train_evaluate_model(df_encoded)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
    st.write("Classification Report:")
    st.write(class_report)
    
    st.header("Decision Tree Visualization")
    visualize_decision_tree(df_encoded)

if __name__ == "__main__":
    main()