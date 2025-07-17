# Audio Visualizer & CNN Classifier

A deep learning-powered audio classification system built with PyTorch and deployed on Modal. This project combines audio signal processing, convolutional neural networks, and real-time inference to classify environmental sounds from the ESC-50 dataset.

## 🎯 What This Project Does

This audio visualizer performs real-time classification of environmental sounds using a custom ResNet-based CNN architecture. It processes audio files, converts them to mel-spectrograms, and provides:

- **Audio Classification**: Identifies 50 different environmental sound categories
- **Feature Map Visualization**: Extracts and visualizes intermediate CNN representations
- **Real-time Inference**: Fast prediction via Modal's cloud infrastructure
- **Waveform Analysis**: Processes and analyzes audio waveforms
- **Training Pipeline**: Complete training setup with data augmentation and monitoring

## ✨ Key Features

- **🧠 Deep Learning Model**: Custom ResNet-based CNN with residual blocks for robust audio classification
- **🎵 Audio Processing**: Advanced mel-spectrogram generation with frequency/time masking augmentation
- **☁️ Cloud Deployment**: Scalable inference using Modal's serverless GPU infrastructure
- **📊 Training Monitoring**: TensorBoard integration for real-time training visualization
- **🔄 Data Augmentation**: Mixup augmentation and SpecAugment for improved model generalization
- **📈 Feature Visualization**: Extract and visualize intermediate CNN feature maps
- **⚡ Fast Inference**: Optimized inference pipeline with batched processing

## 🏗️ Project Structure

```text
audio_visualizer/
├── model.py                    # ResNet-based CNN architecture
├── train.py                    # Training pipeline with Modal integration
├── main.py                     # Inference server and local testing
├── requirements.txt            # Python dependencies
├── audio-cnn-visualizer/       # Frontend application
│   ├── src/                    # React/Next.js source code
│   ├── public/                 # Static assets
│   ├── package.json            # Frontend dependencies
│   └── ...                     # Other frontend files
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+ and npm/yarn (for frontend)
- Modal account ([sign up here](https://modal.com))
- Audio files for testing (supports WAV, MP3, etc.)

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd audio_visualizer
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Modal CLI**

   ```bash
   pip install modal
   ```

5. **Authenticate with Modal**

   ```bash
   modal setup
   ```

6. **Setup Frontend Application**

   ```bash
   # Navigate to frontend directory
   cd audio-cnn-visualizer
   
   # Install frontend dependencies
   npm install
   # or
   yarn install
   ```

## 🔧 Modal Setup

### 1. Training Setup

The training script automatically sets up the ESC-50 dataset and creates necessary Modal volumes:

```bash
# Start training on Modal (uses A10G GPU)
modal run train.py
```

### 2. Inference Setup

Deploy the inference server:

```bash
# Deploy inference endpoint
modal serve main.py
```

### 3. Local Testing

Test the inference with a local audio file:

```bash
# Ensure you have an audio file named 'chirpingbirds.wav' in the project directory
modal run main.py
```

### 4. Frontend Application

Run the web interface to interact with your model:

```bash
# Navigate to frontend directory
cd audio-cnn-visualizer

# Start the development server
npm run dev
# or
yarn dev

# Open your browser to http://localhost:3000
```

## 📊 TensorBoard Monitoring

Monitor training progress with TensorBoard:

### 1. During Training

Training logs are automatically saved to `/models/tensorboard_logs/` in the Modal volume.

### 2. Local TensorBoard Access

To view training logs locally, you need to download them from Modal:

```bash
# First, mount the Modal volume locally (if supported)
# Or access logs through Modal's web interface

# Start TensorBoard locally (if you have logs downloaded)
tensorboard --logdir=./tensorboard_logs
```

### 3. Training Metrics

The following metrics are tracked:

- **Training Loss**: Cross-entropy loss with label smoothing
- **Validation Loss**: Validation set performance
- **Validation Accuracy**: Top-1 accuracy on validation set
- **Learning Rate**: Dynamic learning rate from OneCycleLR scheduler

## 🎮 Usage Examples

### Training a Model

```bash
# Train on Modal with automatic dataset download
modal run train.py
```

### Running Inference

```bash
# Deploy inference server
modal serve main.py

# Test with local audio file
modal run main.py
```

### Running the Web Application

```bash
# Start the frontend (from audio-cnn-visualizer directory)
cd audio-cnn-visualizer
npm run dev

# The web app will be available at http://localhost:3000
# Upload audio files and see real-time classification and visualization
```

### Custom Audio Classification

```python
import base64
import requests
import soundfile as sf
import io

# Load your audio file
audio_data, sample_rate = sf.read("your_audio.wav")

# Convert to base64
buffer = io.BytesIO()
sf.write(buffer, audio_data, sample_rate, format="WAV")
audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

# Send to inference endpoint
payload = {"audio_data": audio_b64}
response = requests.post("YOUR_MODAL_ENDPOINT_URL", json=payload)
result = response.json()

print("Predictions:", result["predictions"])
```

## 🏋️ Model Architecture

- **Base Architecture**: ResNet-inspired CNN with residual blocks
- **Input**: Mel-spectrograms (128 mel bins, 1024 FFT)
- **Layers**: 4 residual block layers with increasing channels (64→128→256→512)
- **Output**: 50-class classification (ESC-50 categories)
- **Regularization**: Dropout (0.5) + Label smoothing (0.1)

## 📈 Training Configuration

- **Dataset**: ESC-50 (2000 samples, 50 classes)
- **Batch Size**: 32
- **Epochs**: 100
- **Optimizer**: AdamW with weight decay (0.01)
- **Scheduler**: OneCycleLR (max_lr=0.002)
- **Augmentation**: Mixup (30% chance) + SpecAugment

## 🐛 Troubleshooting

### Common Issues

1. **Modal Authentication Error**

   ```bash
   modal setup
   ```

2. **GPU Memory Issues**
   - Reduce batch size in `train.py`
   - Use smaller model variant

3. **Audio File Format Issues**
   - Ensure audio files are in supported formats (WAV, MP3, FLAC)
   - Check sample rate compatibility

4. **Inference Endpoint Not Found**
   - Ensure the inference server is deployed: `modal serve main.py`
   - Check Modal dashboard for endpoint status

5. **Frontend Connection Issues**
   - Ensure the Modal inference endpoint is running
   - Update the API endpoint URL in the frontend configuration
   - Check CORS settings if making cross-origin requests

6. **Node.js/npm Issues**
   - Ensure Node.js 18+ is installed
   - Clear npm cache: `npm cache clean --force`
   - Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

- [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50)
- [Modal Documentation](https://modal.com/docs)
- [PyTorch Audio](https://pytorch.org/audio/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
