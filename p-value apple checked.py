from scipy.stats import norm

# Function to calculate the z-statistic and p-value for the change in accuracy
def calculate_p_value(accuracy_before, accuracy_after, n_before, n_after):
    p1 = accuracy_before
    p2 = accuracy_after
    p_pool = (p1 * n_before + p2 * n_after) / (n_before + n_after)
    
    z_stat = (p1 - p2) / (p_pool * (1 - p_pool) * (1 / n_before + 1 / n_after))**0.5
    p_value = norm.sf(abs(z_stat)) * 2  # two-tailed test
    return z_stat, p_value

# Assume the size of the dataset before and after the introduction of errors is the same
n_before = n_after = 100 

# Original and new accuracies
accuracy_before = 0.89  # 89%
accuracy_after = 0.76  # 76%

# Call the function to calculate the p-value
z_stat, p_value = calculate_p_value(accuracy_before, accuracy_after, n_before, n_after)

print(z_stat, p_value)
