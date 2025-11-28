# Missing values
print(df.isnull().sum())

# Target distribution
print(f"\nTarget Distribution:")
print(df['loan_condition_int'].value_counts())
print(f"Default Rate: {df['loan_condition_int'].mean()*100:.2f}%")

def prepare_dataset(df):
    """
    Prepare the dataset for modeling
    """
    print("\n Data Preparation")
    print(f"Initial dataset shape: {df.shape}")

    # Create target variable (is_default)
    if 'is_default' not in df.columns:
        if 'loan_condition' in df.columns:
            df['is_default'] = (df['loan_condition'] == 'Bad Loan').astype(int)

    print(f"Default rate: {df['is_default'].mean():.2%}")

    return df

"""Missing Value Treatment"""

def handle_missing_values(df):
    """
    Identify and impute missing values
    """
    print("\n Missing Value Treatment ")

    # Identify missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    print(f"\nVariables with missing values: {len(missing_pct)}")
    if len(missing_pct) > 0:
        print(missing_pct.head(10))

    # Create missing indicators for important variables
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    # Impute numerical variables with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Impute categorical variables with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df

"""Outlier Detection and Treatment"""

def treat_outliers(df, columns, method='cap', threshold=3):
    """
    Detect and treat outliers using IQR or Z-score method
    """
    print("\n Outlier Treatment ")

    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            if outliers > 0:
                print(f"{col}: {outliers} outliers detected")

                if method == 'cap':
                    # Cap outliers
                    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

    return df

"""WEIGHT OF EVIDENCE (WOE) AND INFORMATION VALUE (IV)"""

def calculate_woe_iv(df, feature, target):
    """
    Calculate Weight of Evidence and Information Value
    """
    # Create bins for continuous variables
    if df[feature].dtype in [np.float64, np.int64]:
        try:
            df[f'{feature}_bin'] = pd.qcut(df[feature], q=5, duplicates='drop')
        except:
            df[f'{feature}_bin'] = pd.cut(df[feature], bins=5, duplicates='drop')
        feature_bin = f'{feature}_bin'
    else:
        feature_bin = feature

    # Calculate WOE and IV
    df_grouped = df.groupby(feature_bin)[target].agg(['count', 'sum'])
    df_grouped.columns = ['Total', 'Bad']
    df_grouped['Good'] = df_grouped['Total'] - df_grouped['Bad']

    total_good = df_grouped['Good'].sum()
    total_bad = df_grouped['Bad'].sum()

    df_grouped['Good_Dist'] = df_grouped['Good'] / total_good
    df_grouped['Bad_Dist'] = df_grouped['Bad'] / total_bad

    # Avoid division by zero
    df_grouped['Good_Dist'] = df_grouped['Good_Dist'].replace(0, 0.0001)
    df_grouped['Bad_Dist'] = df_grouped['Bad_Dist'].replace(0, 0.0001)

    df_grouped['WOE'] = np.log(df_grouped['Good_Dist'] / df_grouped['Bad_Dist'])
    df_grouped['IV'] = (df_grouped['Good_Dist'] - df_grouped['Bad_Dist']) * df_grouped['WOE']

    iv = df_grouped['IV'].sum()

    return df_grouped, iv

def calculate_iv_all_features(df, target, features):
    """
    Calculate IV for all features and rank them
    """
    print("\n--- Information Value Calculation ---")

    iv_dict = {}
    for feature in features:
        try:
            _, iv = calculate_woe_iv(df, feature, target)
            iv_dict[feature] = iv
        except:
            pass

    iv_df = pd.DataFrame(list(iv_dict.items()), columns=['Feature', 'IV'])
    iv_df = iv_df.sort_values('IV', ascending=False)

    # IV interpretation
    def iv_category(iv):
        if iv < 0.02:
            return 'Not Predictive'
        elif iv < 0.1:
            return 'Weak'
        elif iv < 0.3:
            return 'Medium'
        elif iv < 0.5:
            return 'Strong'
        else:
            return 'Suspicious'

    iv_df['Predictive_Power'] = iv_df['IV'].apply(iv_category)

    print("\nTop 10 Features by Information Value:")
    print(iv_df.head(10))

    return iv_df

"""CORRELATION AND MULTICOLLINEARITY CHECK"""

def check_correlation(df, features, threshold=0.7):
    """
    Check correlation and identify highly correlated features
    """
    print("\n Correlation Analysis ")

    numeric_features = df[features].select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_features].corr().abs()

    # Find highly correlated pairs
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_corr_pairs = [
        (column, row, corr_matrix.loc[row, column])
        for column in upper_tri.columns
        for row in upper_tri.index
        if upper_tri.loc[row, column] > threshold
    ]

    if high_corr_pairs:
        print(f"\nHighly correlated pairs (correlation > {threshold}):")
        for col1, col2, corr in high_corr_pairs[:10]:
            print(f"{col1} <-> {col2}: {corr:.3f}")
    else:
        print(f"\nNo highly correlated pairs found (threshold: {threshold})")

    return corr_matrix, high_corr_pairs