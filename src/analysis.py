import matplotlib.pyplot as plt
import seaborn as sns

def run_analysis(df):
    print("\n📊 Basic Info")
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    # Data balance
    print("\n⚖️ Popularity Balance:")
    print(df["popularity_label"].value_counts())
    print("Ratio (Popular/Not Popular):", 
          round(df["popularity_label"].value_counts(normalize=True), 2))

    # Missing values
    print("\n❌ Missing Values per Column:")
    print(df.isnull().sum())

    # Numeric statistics
    print("\n📈 Numeric Description:")
    print(df.describe())

    # Plot balance
    plt.figure(figsize=(6,4))
    sns.countplot(x="popularity_label", data=df)
    plt.title("Distribution of Popular vs Not Popular Songs")
    plt.show()

    # Correlation (numeric columns only)
    numeric_df = df.select_dtypes(include=["number"])
    plt.figure(figsize=(12,6))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features Only)")
    plt.show()
