model = Sequential([Input(shape=(1,)),
                    Dense(56, activation="leaky_relu"), 
                    Dense(1, activation="linear")])
x = Normilize_with_MinMax(data[0]).reshape(-1, 1)
y = Normilize_with_MinMax(data[2]).reshape(-1, 1)
x_val = Normilize_with_MinMax(data[1]).reshape(-1, 1)
y_val = Normilize_with_MinMax(data[3]).reshape(-1, 1)
print(x, y, x_val, y_val )
model.compile(
            optimizer=Adam(),
            loss="mse",
            metrics=["mape"],
        )
callback = EarlyStopping(
                monitor='val_loss',
                patience=10
                )
history = model.fit(
            x = x, 
            y = y,
            epochs=10,
            validation_data=(x_val, y_val),
            callbacks=callback,)