"""Train Test Split"""

def split_data(df, target, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets
    """
    print("\n--- Data Splitting ---")

    # First split: separate test set
    X = df.drop(columns=[target])
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted,
        random_state=random_state, stratify=y_temp
    )

    print(f"Training set: {X_train.shape[0]} samples ({y_train.mean():.2%} default rate)")
    print(f"Validation set: {X_val.shape[0]} samples ({y_val.mean():.2%} default rate)")
    print(f"Test set: {X_test.shape[0]} samples ({y_test.mean():.2%} default rate)")

    return X_train, X_val, X_test, y_train, y_val, y_test

"""LOGISTIC REGRESSION MODEL"""

def build_logistic_model(X_train, y_train, X_val, y_val, selected_features):
    """
    Build and train logistic regression model
    """
    print("\n--- Logistic Regression Model ---")

    # Select only the chosen features
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]

    # Build model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train_selected, y_train)

    # Predictions
    y_train_pred_proba = model.predict_proba(X_train_selected)[:, 1]
    y_val_pred_proba = model.predict_proba(X_val_selected)[:, 1]

    # Calculate AUC
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)

    print(f"Training AUC: {train_auc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Gini Coefficient (Train): {2*train_auc - 1:.4f}")
    print(f"Gini Coefficient (Validation): {2*val_auc - 1:.4f}")

    # Feature coefficients
    coef_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', ascending=False)

    print("\nFeature Coefficients:")
    print(coef_df)

    return model, train_auc, val_auc

"""CONVERT PD TO CREDIT SCORE"""

def convert_to_scorecard(model, feature_names, pdo=20, base_score=600, base_odds=50):
    """
    Convert logistic regression model to credit scorecard
    PDO = Points to Double the Odds
    Base Score = Score at base odds
    Base Odds = Odds at base score (e.g., 50:1 means 2% default rate)
    """
    print("\n--- Scorecard Conversion ---")

    # Calculate factor and offset
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    print(f"Factor: {factor:.2f}")
    print(f"Offset: {offset:.2f}")

    # Convert coefficients to points
    intercept_points = offset - factor * model.intercept_[0]

    scorecard = pd.DataFrame({
        'Feature': ['Intercept'] + list(feature_names),
        'Coefficient': [model.intercept_[0]] + list(model.coef_[0]),
        'Points': [intercept_points] + list(-factor * model.coef_[0])
    })

    print("\nScorecard Points:")
    print(scorecard)

    return scorecard, factor, offset

def calculate_scores(X, model, factor, offset):
    """
    Calculate credit scores for dataset
    """
    # Get log odds
    log_odds = model.decision_function(X)

    # Convert to scores
    scores = offset - factor * log_odds

    return scores