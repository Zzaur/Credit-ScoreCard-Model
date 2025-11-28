"""KOLMOGOROV-SMIRNOV (KS) STATISTIC"""

def calculate_ks_statistic(y_true, y_pred_proba):
    """
    Calculate KS statistic
    """
    # Sort by predicted probability
    df_ks = pd.DataFrame({
        'y_true': y_true,
        'y_pred_proba': y_pred_proba
    }).sort_values('y_pred_proba', ascending=False)

    # Calculate cumulative distributions
    df_ks['bad_cumsum'] = df_ks['y_true'].cumsum()
    df_ks['good_cumsum'] = (1 - df_ks['y_true']).cumsum()

    total_bad = df_ks['y_true'].sum()
    total_good = len(df_ks) - total_bad

    df_ks['bad_rate'] = df_ks['bad_cumsum'] / total_bad
    df_ks['good_rate'] = df_ks['good_cumsum'] / total_good

    # KS statistic
    df_ks['ks'] = df_ks['bad_rate'] - df_ks['good_rate']
    ks_stat = df_ks['ks'].max()

    return ks_stat

"""MODEL PERFORMANCE METRICS"""

def evaluate_model_performance(y_true, y_pred_proba, y_scores=None):
    """
    Comprehensive model evaluation
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*80)

    # AUC and Gini
    auc = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc - 1

    print(f"\nAUC (Area Under Curve): {auc:.4f}")
    print(f"Gini Coefficient: {gini:.4f}")

    # KS Statistic
    ks_stat = calculate_ks_statistic(y_true, y_pred_proba)
    print(f"KS Statistic: {ks_stat:.4f}")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Score distribution
    if y_scores is not None:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(y_scores[y_true == 0], bins=50, alpha=0.7, label='Good', color='green')
        plt.hist(y_scores[y_true == 1], bins=50, alpha=0.7, label='Bad', color='red')
        plt.xlabel('Credit Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution by Loan Status')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        score_bins = pd.cut(y_scores, bins=10)
        default_rate_by_score = pd.DataFrame({
            'score_bin': score_bins,
            'default': y_true
        }).groupby('score_bin')['default'].mean()

        default_rate_by_score.plot(kind='bar', color='steelblue')
        plt.xlabel('Score Bin')
        plt.ylabel('Default Rate')
        plt.title('Default Rate by Score Bin')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return auc, gini, ks_stat

"""POPULATION STABILITY INDEX (PSI)"""

def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index
    """
    def scale_range(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val)

    # Scale to 0-1
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    expected_scaled = scale_range(expected, min_val, max_val)
    actual_scaled = scale_range(actual, min_val, max_val)

    # Create bins
    breakpoints = np.linspace(0, 1, bins + 1)

    expected_counts = np.histogram(expected_scaled, bins=breakpoints)[0]
    actual_counts = np.histogram(actual_scaled, bins=breakpoints)[0]

    # Calculate percentages
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = psi_values.sum()

    print(f"\nPopulation Stability Index (PSI): {psi:.4f}")

    if psi < 0.1:
        print("Interpretation: Population is STABLE")
    elif psi < 0.25:
        print("Interpretation: Population has SHIFTED (monitor closely)")
    else:
        print("Interpretation: Population has SIGNIFICANTLY SHIFTED (recalibrate model)")

    return psi