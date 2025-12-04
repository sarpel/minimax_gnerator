# Training Guide

Once you have generated and augmented your dataset, the next step is to train a wake word model. WakeGen is designed to work seamlessly with **OpenWakeWord**.

## 1. Exporting the Dataset

First, we need to organize the data into a format OpenWakeWord understands.

```bash
wakegen export --data-dir ./output --format openwakeword --output-path ./training_data
```

This will create a folder structure with:
*   Positive samples (your wake word)
*   Negative samples (background noise, other speech)
*   A JSON manifest file describing the data.

## 2. Generating a Training Script

OpenWakeWord training can be complex. WakeGen simplifies this by generating a training script for you.

```bash
wakegen train-script --model-type openwakeword --output-script train.sh
```

This creates a `train.sh` file with all the necessary commands and hyperparameters.

## 3. Running the Training

You will need to install the `openwakeword` library separately (it's not a direct dependency of WakeGen to keep things light).

```bash
pip install openwakeword
```

Then, run the generated script:

```bash
bash train.sh
```

## 4. Testing the Model

After training, you will get a `.tflite` or `.onnx` model file. You can test it using WakeGen's validation tool:

```bash
wakegen validate --model path/to/model.tflite --test-dir ./test_data
```

This will report the False Accept Rate (FAR) and False Reject Rate (FRR).

## Tips for Better Training

*   **More Data**: 100 samples is a minimum. 1000+ is better.
*   **Diverse Voices**: Use multiple TTS providers (Edge, Piper, Minimax) to get different voices.
*   **Heavy Augmentation**: Don't be afraid to add lots of noise. Real life is noisy!