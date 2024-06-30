# conjoint-analysis-

%cd /content/drive/MyDrive/Conjoint Analysis/Conjoint Analysis

# Install library
!pip install squarify

#Import Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import squarify

#Load the data
df = pd.read_csv('netflix_customer_survey.csv')
df.head()

# Data Processing

# Isolate X and Y
y = df.selected
x = df.drop(columns=['selected', 'customerid'])
x.head(2)

# Create dummy variables
x_dummy = pd.get_dummies(x, columns = x.columns)
x_dummy.head()

# Regression Model

# Build the regression model
model = sm.GLM(y,
               x_dummy,
               family = sm.families.Binomial()).fit()
model.summary()

# Conjoint Analysis

# Create a DF with the results
df_result = pd.DataFrame({'param_name': model.params.keys(),
                          'partworth': model.params.values,
                          'pval': model.pvalues
                          })
df_result

# Identifying the statistically significant variables
df_result['is_significant'] = df_result['pval'] < 0.05

df_result['color'] = ['blue' if x else 'red' for x in df_result['is_significant']]


df_result.head()

# Sort values
df_result = df_result.sort_values(by = "partworth", ascending = True)

# Plot the Partworth
f, ax = plt.subplots(figsize = (14,10))
values = df_result.partworth
xbar = np.arange(len(values))
plt.title("Perceived value for customers", fontsize = 25)

# Bar plot
plt.barh(xbar,
         values,
         color = df_result['color'])
#Customizing the ticks
plt.yticks(xbar,
           labels = df_result['param_name'],
           fontsize = 20)
plt.xticks(fontsize = 20)
plt.show()

#Specific groups of features


# Isolate the feature group
feature = "price"
attributes = []
coefficients = []
for i in range(len(df_result)):
  if df_result.iloc[i,0].find(feature) == 0:
    attributes.append(df_result.iloc[i,0])
    coefficients.append(df_result.iloc[i,1])


# Lollipop chart
# Plot the Partworth
f, ax = plt.subplots(figsize = (10,4))
plt.title("Perceived value per feature group", fontsize = 18)

# Lollipop plot
(markers, stemlines, baseline) = plt.stem(attributes,
                                          coefficients,
                                          linefmt = "-",
                                          markerfmt = "o",
                                          basefmt = " ")

# Customize the lollipop
plt.setp(stemlines,
         color = "skyblue",
         linewidth = 4)

plt.setp(markers,
         color = "black",
         markersize = 10)


#Customizing the ticks
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16,
           rotation = 30)
#Plot
plt.show()

# Feature Importance

# Coefficients
features_partworth = {}
for key, coefficient in model.params.items():
  split_key = key.split('_')
  feature = split_key[0]
  if feature not in features_partworth:
    features_partworth[feature] = list()
  features_partworth[feature].append(coefficient)
features_partworth

# Calculation: maximum of a feature(price_8) minus minimum of a feature (price_20)
importance_per_feature = {k: max(v) - min(v) for k, v in features_partworth.items()}
importance_per_feature

# Computing the sum of importances
total_importances = sum(importance_per_feature.values())
total_importances

# Relative importance (adds up to 100)
relative_importance_per_feature = {
    k: round(100 * v/total_importances,1) for k, v in importance_per_feature.items()
}
relative_importance_per_feature

# Build DF
df_importances = pd.DataFrame(
    list(relative_importance_per_feature.items()),
    columns = ['feature', 'relative_importance'])
df_importances

# Treemap
squarify.plot(sizes = df_importances.relative_importance,
              label = df_importances.feature,
              color = ["red", "yellow", "pink", "orange"],
              value = df_importances.relative_importance,
              alpha = 0.8,
              pad = 1)
plt.axis("off")
plt.show()

# Interaction Terms


# Create Interaction terms
df['content_ads'] = df.ExtraContent + "_" + df.ads
df.head()

# Remove the vars in the interaction term
df_interaction = df.drop(columns = ['ExtraContent', 'ads'])
df_interaction.head()

y = df_interaction.selected
x = df_interaction.drop(columns = ['selected', 'customerid'])
x.head(2)

#create dummy variables
x_dummy = pd.get_dummies(x, columns = x.columns)
x_dummy.head()

# Regression Model
model2 = sm.GLM(y,
                x_dummy,
                family = sm.families.Binomial()).fit()
model2.summary()
