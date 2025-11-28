# Gesture Research

## Download
Clone the repo and download the three zip files in the releases tab to the right.
```
git clone https://github.com/cbrt-1/gestures-research.git
```
Cd into the gestures-research folder.
Make a folder called data/processed. The training script will look for files in here.
```
mkdir -p data/processed
```
Unzip all three zip files into this new folder.

## Training

### Custom Dataset
To add custom data(ex: snapping fingers), run:
```
uv run scripts/custom_trainingset.py
```
It will open the camera and live record all the 138 dimension mediapipe output. Press `r` to record. Press `s` to save the recording. Press `q` to quit the program.
The data is not labeled, meaning that the recording can capture multiple gestures. Ideally the custom dataset is long containing multiple gestures with both the right and left hands.

### Start Training
To start training, run:
```
uv run src/train_model.py
```
It should load all the files from the data/processed folder and train off of it.
Some tuning of hyperparameters might be needed, or modifying the training script (such as adding some jitter).
The vq-vae model might also need to be changed if it turns out not large enough or too large leading to overfit. The file can be found in `src/model/vqvae.py`.
A large perplexity score (% of code book being used) is good.
Resetting dead vectors is also good.

### Visualizing the Output
Run the visualization script to see the outputted tokens of the model. It doesn't have to be a perfect or good reconstruction. 
The goal is to have distinct gestures to map to distinct sequences of tokens. If the tokens for static fist is different from a wave or a shaking fist, it was successful.
```
uv run src/visualize.py # live visualization (probably more useful)
uv run src/dataset_visualize # view reconstructions from the dataset. **Didnt get an opportunity to test this.**
```
## Encoding Scheme
A 138 dimesional vector is computed from mediapipe hands.

More information [here](encoding.md).

