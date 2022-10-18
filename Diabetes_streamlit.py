import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('/Users/chidex/diabetes_data.sav', 'rb'))


input_data = (1,85,66,29,0,26.6,0.351,31) # 0

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('This patient is not diabetic')
else:
    print('This patient is diabetic')