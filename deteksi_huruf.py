import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def min_max_scaling(data):
    min = 0
    max = 255

    return (data-min) / (max - min)

def save_model(means, variances, p, classes):
    means_df = pd.DataFrame(means, index=classes)
    means_df.to_csv('data/means.csv')
    
    variances_df = pd.DataFrame(variances, index=classes)
    variances_df.to_csv('data/variances.csv')
    
    priors_df = pd.DataFrame({
        'prior': p,
        'class': classes
    })
    priors_df.to_csv('data/priors_class.csv', index=False)
    

def load_model():
    means_df = pd.read_csv('data/means.csv', index_col=0)
    variances_df = pd.read_csv('data/variances.csv', index_col=0)
    priors_df = pd.read_csv('data/priors_class.csv')
    
    means = means_df.to_numpy()
    variances = variances_df.to_numpy()
    classes = priors_df['class'].tolist()
    p = priors_df['prior'].to_numpy()
    
    return means, variances, p, classes

# Read data
def read_data():
    url = 'data/handwritten_data_785.csv'
    df = pd.read_csv(url, header=None)

    return df

def read_data_test():
    url = 'data/test.csv'
    df = pd.read_csv(url, header=None)

    return df
# -----------------------------------------------------------

# Split training dan testing data
def split_data(df):
    split_index = int(0.8 * len(df))

    df_randomized = df.sample(frac = 1, random_state=2)
    df_randomized_normalized = df_randomized

    df_train = df_randomized_normalized.iloc[:split_index]
    df_test = df_randomized_normalized.iloc[split_index:]

    label_train = df_train.iloc[:, 0].apply(lambda x: chr(x + ord('A')))
    feature_train = min_max_scaling(df_train.iloc[:, 1:]).replace({0: 1e-12})

    label_test = df_test.iloc[:, 0].apply(lambda x: chr(x + ord('A')))
    feature_test = min_max_scaling(df_test.iloc[:, 1:]).replace({0: 1e-12})

    return label_train, feature_train, label_test, feature_test
# -----------------------------------------------------------

# Mean, variance, and label probability
def model_train(label_train, feature_train): 
    means = feature_train.groupby(label_train).mean().to_numpy()
    variances = feature_train.groupby(label_train).var(ddof=0).to_numpy()

    variances += 1e-9

    p = label_train.value_counts(normalize=True).sort_index().to_numpy()
    classes = []

    for x in range(len(p)):
        classes.append(chr(ord('A')+ x))

    save_model(means, variances, p, classes)

    return means, variances, p, classes
# -----------------------------------------------------------

# Gaussian function for continous data 
def gaussian_function(x, var, mean): 
    return 1 / (np.sqrt(2*np.pi*var)) * np.exp(-np.power(x - mean, 2) / (2 * var))
# -----------------------------------------------------------

# Predict one sample with probabilities
    # Can't use because of overflow
    def predict_single_sample(sample, means, variances, p, classes):
        total_classes = len(classes)
        class_probabilities = np.zeros(total_classes)
        
        for i in range(total_classes):
            likelihood = gaussian_function(sample, variances[i], means[i])

            total_likelihood = 1
            for x in likelihood:
                total_likelihood *= x
            
            class_probabilities[i] = total_likelihood * p[i]
        
        return class_probabilities

# Use log probabilities to prevent overflow
def predict_single_sample_log(sample, means, variances, p, classes):
    total_classes = len(classes)
    class_probabilities = np.zeros(total_classes)
    
    for i in range(total_classes):
        likelihood = gaussian_function(sample, variances[i], means[i])
        likelihood += 1e-300
        
        log_likelihood = np.sum(np.log(likelihood))

        log_p = np.log(p[i] + 1e-300)
        
        class_probabilities[i] = log_likelihood + log_p
    
    # From log to normalized probability
    max_log_prob = np.max(class_probabilities)
    class_probabilities = class_probabilities - max_log_prob
    class_probabilities = np.exp(class_probabilities)
    
    prob_sum = np.sum(class_probabilities)
    if prob_sum > 0:
        class_probabilities = class_probabilities / prob_sum
    else:
        class_probabilities = np.ones(total_classes) / total_classes
    
    return class_probabilities
# -----------------------------------------------------------

# Predict dataset with probability
def predict_multiple_sample(feature_test, means, variances, p, classes):
    
    predictions = []
    probabilities = []
    
    for i in range(len(feature_test)):
        sample = feature_test.iloc[i].values
        
        class_probabilities = predict_single_sample_log(sample, means, variances, p, classes)
        
        if np.sum(class_probabilities) > 0:
            class_probabilities = class_probabilities / np.sum(class_probabilities)
        
        probabilities.append(class_probabilities)
        
        predicted_class_index = np.argmax(class_probabilities)
        predicted_class = classes[predicted_class_index] 
        predictions.append(predicted_class)
    
    return np.array(predictions), np.array(probabilities)
# -----------------------------------------------------------

# Calculate Accuracy 
def manual_accuracy_score(true, predict):
    correct = 0
    total = len(true)
    
    true = true.values

    for i in range(total):
        if true[i] == predict[i]: 
            correct += 1
    
    return correct / total
# -----------------------------------------------------------

# Create confusion matrix
def create_confusion_matrix(true, predict, classes):
    true = true.values
    n_classes = len(classes)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for true_label, pred_label in zip(true, predict):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        confusion_matrix[true_idx][pred_idx] += 1
    
    return confusion_matrix
# -----------------------------------------------------------

# Display confusion matrix
def display_confusion_matrix(confusion_matrix, classes):
    
    df = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
    df.index.name = 'True Alphabet'
    df.columns.name = 'Predicted Alphabet'
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=0.5)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Alphabet', fontsize=12)
    plt.xlabel('Predicted Alphabet', fontsize=12)
    
    plt.tight_layout()
    plt.show()
# -----------------------------------------------------------

# Main to train model
def main_1():
    df = read_data()

    label_train, feature_train, label_test, feature_test = split_data(df)
    means, variances, p, classes = model_train(label_train, feature_train)

    predictions, probabilities = predict_multiple_sample(feature_test, means, variances, p, classes)

    accuracy = manual_accuracy_score(label_test, predictions)
    print(f"Accuracy: {accuracy}")
    
    return accuracy, predictions, probabilities

# Main to test model
def main_2(): 
    df = read_data_test()

    label_train, feature_train, label_test, feature_test = split_data(df)
    means, variances, p, classes = load_model()
    predictions, probabilities = predict_multiple_sample(feature_test, means, variances, p, classes)

    accuracy = manual_accuracy_score(label_test, predictions)
    print(f"Accuracy: {accuracy}")
    
    confusion_matrix = create_confusion_matrix(label_test, predictions, classes)
    display_confusion_matrix(confusion_matrix, classes)

    return accuracy, predictions, probabilities

main_2()