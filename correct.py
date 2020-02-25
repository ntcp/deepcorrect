from txt2txt import build_model, infer

model, params = build_model(params_path='params', enc_lstm_units=256)
model.load_weights('checkpoint')

while 1:
    print('Enter input')
    print(infer(input(), model, params))



