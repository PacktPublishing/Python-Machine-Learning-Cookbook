from sklearn.datasets import samples_generator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

# generate sample data
X, y = samples_generator.make_classification(
        n_informative=4, n_features=20, n_redundant=0, random_state=5)

# Feature selector 
selector_k_best = SelectKBest(f_regression, k=10)

# Random forest classifier
classifier = RandomForestClassifier(n_estimators=50, max_depth=4)

# Build the machine learning pipeline
pipeline_classifier = Pipeline([('selector', selector_k_best), ('rf', classifier)])

# We can set the parameters using the names we assigned
# earlier. For example, if we want to set 'k' to 6 in the
# feature selector and set 'n_estimators' in the Random 
# Forest Classifier to 25, we can do it as shown below
pipeline_classifier.set_params(selector__k=6, 
        rf__n_estimators=25)

# Training the classifier
pipeline_classifier.fit(X, y)

# Predict the output
prediction = pipeline_classifier.predict(X)
print "\nPredictions:\n", prediction

# Print score
print "\nScore:", pipeline_classifier.score(X, y)                        

# Print the selected features chosen by the selector
features_status = pipeline_classifier.named_steps['selector'].get_support()
selected_features = []
for count, item in enumerate(features_status):
    if item:
        selected_features.append(count)

print "\nSelected features (0-indexed):", ', '.join([str(x) for x in selected_features])

