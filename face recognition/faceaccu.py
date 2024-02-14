from sklearn.metrics import accuracy_score

# Assuming X_test and y_test are the test images and their true labels, respectively
# Assuming clf is your trained classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
