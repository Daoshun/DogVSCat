from keras import layers
from keras import models
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import np_utils
from keras.optimizers import Adam

train_dir = r'/home/vincent/data/dogsvscats/cats_and_dogs_small/train'
validation_dir = r'/home/vincent/data/dogsvscats/cats_and_dogs_small/validation'
test_dir = r'/home/vincent/data/dogsvscats/cats_and_dogs_small/test'

# 调整像素值
# train_datagen = ImageDataGenerator(rescale=1./255)
# validation_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=40, # 0~180的度数，指定随机选择的的角度
    width_shift_range=0.2,
    height_shift_range=0.2, # 用来指定水平和竖直方向随机移动的程度，0～1
    rescale=1./255, # RGB0~255 参数太高对于模型去处理，给一个典型的learning rate ，so固定在0～1的值
    shear_range=0.2, # 剪切变换的程度
    zoom_range=0.2, #随机的放大
    horizontal_flip=True, # 对图片进行水平翻转
    fill_mode='nearest' #对需要的地方进行像素填充
)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
# validation_generator = test_datagen.flow_from_directory(directory=validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
# validation_generator = validation_datagen.flow_from_directory(directory=validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
test_generator = test_datagen.flow_from_directory(directory=test_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
# test_data=test_generator, validation_steps=50

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPool2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPool2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPool2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPool2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# # 二类问题多用sigmoid函数，-1到1,多类问题用softmax强化概率大的值
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
# # model.add(layers.Dense(1, activation='softmax'))
# # model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
# history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50)
#
# model.save('CNN-DataAugmentation.h5')

#载入模型
model = load_model('CNN-DataAugmentation.h5')
result = model.evaluate_generator(test_generator, steps=500, verbose=1)
print(model.metrics_names)
# epochs = range(1,)
print('\nTest Acc', result)
#print('')
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()
