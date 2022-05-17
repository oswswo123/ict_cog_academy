#%% kerastuner

import kerastuner
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train, X_test = X_train / 255 , X_test / 255

#%%

def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    units = hp.Int('units', min_value=32, max_value=512, step=16)
    model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    learning = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )
    return model

# best model 찾기
tuner = kerastuner.Hyperband(build_model, 'val_accuracy', 20)
tuner.search(X_train, y_train, epochs=20, validation_split=0.3)

#%%

best = tuner.get_best_hyperparameters()

# best model로 model build
best_model = tuner.hypermodel.build(best[0])

print(best_model.summary())

#%%

X_train, X_test = X_train.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1)

def build_model2(hp):
    input_layer = tf.keras.Input(shape=(28, 28, 1))
    
    for i in range(hp.Int('n_layers', 1, 10)):
        x = tf.keras.layers.Conv2D(
            hp.Int('filters_'+str(i), 4, 64, step=8), (3, 3), activation='relu')(input_layer)
        
    if (hp.Choice('global_pooling', ['max', 'avg'])) == 'avg':
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        
    x = tf.keras.layers.Dense(
        units=hp.Int('units', min_value=16, max_value=64, step=8),
        activation='relu')(x)
    x = tf.keras.layers.Dropout(
        hp.Choice('dropout_rate', values=[0.2, 0.5]))(x)
    output_layer = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    model2 = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model2.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model2

#%%
    
tuner2 = kerastuner.RandomSearch(build_model2, 'val_accuracy', max_trials=2)
print(tuner2.search_space_summary())
tuner2.search(X_train, y_train, epochs=5, validation_split=0.2)

tuner2.results_summary()

best = tuner2.get_best_hyperparameters()
best_model2 = tuner2.hypermodel.build(best[0])

#%% transfer learning에 kerastuner 사용

from kerastuner.tuners import Hyperband
from kerastuner.applications import HyperResNet

model = HyperResNet(input_shape=(28, 28, 1), classes=10)
tuner3 = Hyperband(model, 'val_accuracy', max_epochs=10)

tuner3.search(X_train, y_train, validation_spite=0.2)

tuner3.results_summary()

best = tuner3.get_best_hyperparameters()
best_model3 = tuner3.hypermodel.build(best[0])

#%%
#%% Training tuning

import tensorflow as tf

a = tf.Variable([2, 3, 4], dtype=tf.float32)

# GradientTape은 Gradient 연산 결과값을 기록함
# 이를 미분하면 Gradient의 방향을 알 수 있기 때문
# Backpropagation의 근간
with tf.GradientTape() as t:
    y = tf.multiply(a, a)

# y = a²,  y' = 2a
print(t.gradient(y, a))

#%% 간단하게 만들어보기

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

loss = tf.keras.losses.SparseCategoricalCrossentropy()

with tf.GradientTape() as tape:
    prediction = model(X_train)
    losses = loss(y_train, prediction)
t.gradient(losses, model.trainable_variables)