# Image-Forgery-Detection

A model trained on the COVERAGE dataset to highlight areas of the image that it believes to be forged.

# Dependencies

```python
pip3 install -r requirements.txt
```

# Models

There are two models, a CNN implementation and an LSTM implementation for detecting forgeries. they can be found in the src folder.

## CNN

To train the CNN model, run the following command.

```python3
python3 /src/CNN/cnn.py
```

## LSTM

To train the LSTM model, run the following command.

```python3
python3 /src/CNN/run_lstm.py
```
