# Xray Classification Project using a Neural Network

### We 're gonna use a Xray Images from various Kaggle datasets to train a neural network to be able to predict if a patient has Pneumonia , Covid or his X-ray is Normal.
<br/>

#  _Installation_


### Use pip to install all necessary packages
- pip install tensorflow
- pip install numpy
- pip install matplotlib

<br/>

# _The Code_
### We create an ImageDataGenerator object for data augmentation purposes and then we load the data from their respective directories(train and validation), using the gray scale as color_mode because they are X-rays.
<br/> 

```python
data_generator = ImageDataGenerator(rescale=1.0/255, zoom_range=0.1, rotation_range=27, width_shift_range=0.05, height_shift_range=0.05)

training_iterator = data_generator.flow_from_directory('train',color_mode='grayscale', class_mode='categorical')

validation_iterator = data_generator.flow_from_directory('test',color_mode='grayscale', class_mode='categorical')
```
<br/>    

### Inside the create_model function as the name implyes we create the Neural Network model with a Convolutional layer to process the image inputs, then some Dropout and Max-Pooling layers to reduce overfitting and parameters and lastly a Flatten and a Dense Layer to reshape and get the output of the NN.

<br/> 

```python
def create_model():
    model = Sequential()
    model.add(layers.Input(shape=training_iterator.image_shape))
    model.add(layers.Conv2D(5, 5, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D(pool_size=(5,5), strides=5))
    model.add(layers.Dropout(0.2))

    model.add(layers.MaxPooling2D(pool_size=(5,5), strides=3))
    model.add(layers.Flatten())
    model.add(layers.Dense(3,activation='softmax'))
    return model
```
<br/>    

### Next, its the test_model function which tests the model's predictive ability in a sample of 10 X-rays from the Validation data.It prints the Actual and the Predicted class for each X-ray was picked as a sample and lastly the overall accuracy across all 10 predictions.
<br/> 

```python
def test_model(model,validation_iterator):
        test_features,test_labels = validation_iterator.next()
        random_indicies = np.random.choice(len(test_features),10)
        test_features = test_features[random_indicies]
        test_labels = test_labels[random_indicies]
        class_names = {0:'Covid',
                      1:"Normal",
                      2:'Pneumonia'}
        test_prediction = model.predict(test_features)
        
        for i,(image, prediction, label) in enumerate(zip(test_features, test_prediction, test_labels)):

            image_name = "X_ray {}".format(i)

            #Gets predicted class according to the highest probability
            predicted_class = np.argmax(prediction)
            #Gets correct label
            actual_class = np.argmax(label)
            

            print(image_name)
            print("\tModel prediction: {}".format(prediction))
            print("\tTrue label: {} ({})".format(class_names[actual_class], actual_class))
            print("\tCorrect:", predicted_class == actual_class)
            #Saves image file using matplotlib
            sample_image = image
            blank_axes(plt.imshow(
                sample_image[:, :, 0],
                cmap = "gray"
            ))
            plt.title(" Predicted: {}, Actual: {}".format(class_names[predicted_class], class_names[actual_class]))
            plt.tight_layout()
            plt.show()

            plt.clf()
          
        print(model.evaluate(test_features,test_labels))
```

<br/>    

### On the last code block we call the model_create function,compile the model using Adam optimizer for the learning rate optimization and categorical_crossentropy as the loss function due to the project being a classification problem. We used an 12 epochs which seems enough to wield a model with almost 94% Accuracy on the Validation Set and training more didnt really seem like granting more benefitial results. Lastly we call the test_model function to print out the results on the sample of 10 X-rays.
<br/>

```python
#Model Creation and Compiling
model = create_model()
model.compile(loss='categorical_crossentropy', optimizer= keras.optimizers.Adam(learning_rate = 0.005), metrics=['acc'])
print(model.summary())
es = EarlyStopping(monitor='val_loss', patience=6)
history = model.fit(training_iterator, epochs=12,batch_size = 20,validation_data=validation_iterator)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss','val_loss'])

test_model(model,validation_iterator)
```

