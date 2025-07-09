import preprocess

df = preprocess.load_data('heart.csv')
df = preprocess.clean_data(df)
df = preprocess.encode_features(df)
X_train, X_test, y_train, y_test = preprocess.split_and_scale(df)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)