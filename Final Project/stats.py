from scipy import stats
# Accuracy scores from 5-fold CV for two models
model_A = [0.90196078,0.8627451, 0.94, 0.92, 0.98]  # RFE+NB
model_B = [0.90196078, 0.8627451, 0.88, 0.92, 0.94]  # SVM
t_stat, p_value = stats.ttest_rel(model_A, model_B)
print(f"p-value: {p_value:.4f}")