# Audio CNN Visualizer

A comprehensive deep learning system for environmental sound classification built with PyTorch and deployed on Modal. This project features a custom ResNet-based CNN architecture for classifying 50 different environmental sound categories from the ESC-50 dataset, along with a modern Next.js web interface for real-time audio analysis and visualization.

## 🎯 What This Project Does

This system performs intelligent environmental sound classification using advanced deep learning techniques. It processes audio files, converts them to mel-spectrograms, and provides:

- **🎵 Audio Classification**: Identifies 50 different environmental sound categories with confidence scores
- **🧠 Feature Map Visualization**: Extracts and visualizes intermediate CNN layer representations
- **☁️ Cloud Inference**: Scalable real-time prediction via Modal's GPU infrastructure  
- **📊 Waveform Analysis**: Comprehensive audio signal processing and visualization
- **🚀 Modern Web Interface**: Interactive Next.js frontend with real-time audio upload and visualization
- **📈 Training Pipeline**: Complete MLOps pipeline with TensorBoard monitoring and data augmentation

## ✨ Key Features

- **🧠 Advanced CNN Architecture**: Custom ResNet-based model with residual blocks and feature map extraction
- **🎵 Sophisticated Audio Processing**: Mel-spectrogram generation with SpecAugment and Mixup augmentation
- **☁️ Serverless Deployment**: Auto-scaling inference using Modal's GPU infrastructure (A10G)
- **🌐 Modern Web Interface**: Interactive Next.js application with TypeScript and Tailwind CSS
- **📊 Training Monitoring**: Real-time TensorBoard integration with comprehensive metrics tracking
- **🔄 Data Augmentation**: Advanced techniques including frequency/time masking and mixup
- **📈 Feature Visualization**: Multi-layer CNN feature map extraction and visualization
- **⚡ Optimized Inference**: Fast prediction pipeline with automatic audio preprocessing
- **🎛️ Audio Format Support**: Compatible with WAV, MP3, FLAC and other common formats

## 🏗️ Project Structure

```text
audio_visualizer/
├── main.py                     # Modal inference server with FastAPI endpoints
├── model.py                    # ResNet-based CNN architecture with residual blocks
├── train.py                    # Complete training pipeline with Modal integration
├── requirements.txt            # Python dependencies (torch, librosa, modal, etc.)
├── chirpingbirds.wav          # Sample audio file for testing
├── tensorboard_logs/          # Training logs and metrics
│   ├── run_20250716_200823/
│   ├── run_20250717_141137/
│   └── run_20250717_153812/
└── audio-cnn-visualizer/      # Next.js frontend application
    ├── src/                   # TypeScript source code
    │   ├── app/              # Next.js app router pages
    │   ├── components/       # React components (Waveform, FeatureMap, etc.)
    │   │   ├── ui/          # Reusable UI components (shadcn/ui)
    │   │   ├── ColorScale.tsx
    │   │   ├── FeatureMap.tsx
    │   │   ├── LoadingAnimation.tsx
    │   │   ├── SuccessAnimation.tsx
    │   │   └── Waveform.tsx
    │   ├── lib/              # Utility functions and colors
    │   └── styles/           # Global CSS styles
    ├── public/               # Static assets
    
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 18+** and npm/yarn for the frontend
- **Modal account** ([sign up here](https://modal.com)) for cloud deployment
- **Audio files** for testing (WAV, MP3, FLAC supported)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Amartya-007/audio_visualizer.git
   cd audio_visualizer
   ```

2. **Set up Python environment**

   ```powershell
   # Create virtual environment
   python -m venv .venv
   
   # Activate on Windows
   .venv\Scripts\Activate.ps1
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Install and configure Modal**

   ```powershell
   # Install Modal CLI
   pip install modal
   
   # Authenticate with Modal
   modal setup
   ```

4. **Set up the frontend application**

   ```powershell
   # Navigate to frontend directory
   cd audio-cnn-visualizer
   
   # Install dependencies
   npm install
   
   # Return to root directory
   cd ..
   ```

## 🔧 Usage

### 1. Training the Model

Train the CNN model on the ESC-50 dataset using Modal's cloud infrastructure:

```powershell
# Start training (automatically downloads ESC-50 dataset)
modal run train.py
```

**Training features:**

- Automatic ESC-50 dataset download and preprocessing
- A10G GPU acceleration for fast training
- TensorBoard logging with real-time metrics
- Model checkpointing and volume persistence
- Advanced data augmentation (Mixup, SpecAugment)

### 2. Deploy Inference Server

Deploy the FastAPI inference endpoint to Modal:

```powershell
# Deploy the inference server
modal serve main.py
```

This creates a scalable inference endpoint that automatically handles:

- Audio file preprocessing and resampling
- Mel-spectrogram generation
- CNN inference with feature map extraction
- Response formatting with predictions and visualizations

### 3. Test Local Inference

Test the deployed model with a sample audio file:

```powershell
# Test with the included sample file
modal run main.py
```

**Expected output:**

```text
First 10 values: [0.0234, -0.0156, 0.0087, ...]
Duration: 5.0
Top predictions:
  - Birds chirping 92.4%
  - Wind 8.21%
  - Water 4.36%
```

### 4. Run the Web Interface

Launch the Next.js frontend for interactive audio analysis:

```powershell
# Navigate to frontend directory
cd audio-cnn-visualizer

# Start development server
npm run dev

# Open http://localhost:3000 in your browser
```

**Web interface features:**

- choose audio files for classification
- Real-time waveform visualization
- CNN feature map display
- Interactive prediction results
- Responsive design with Tailwind CSS

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

### Custom Audio Classification API

```python
import base64
import requests
import soundfile as sf
import io

# Load your audio file
audio_data, sample_rate = sf.read("your_audio.wav")

# Convert to base64 for API transmission
buffer = io.BytesIO()
sf.write(buffer, audio_data, sample_rate, format="WAV")
audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

# Send inference request
payload = {"audio_data": audio_b64}
response = requests.post("YOUR_MODAL_ENDPOINT_URL", json=payload)
result = response.json()

# Parse results
predictions = result["predictions"]
feature_maps = result["visualization"]
spectrogram = result["input_spectrogram"]
waveform = result["waveform"]

print("Top 3 Predictions:")
for pred in predictions:
    print(f"  {pred['class']}: {pred['confidence']:.2%}")
```

### Response Format

```json
{
  "predictions": [
    {"class": "Birds", "confidence": 0.8743},
    {"class": "Wind", "confidence": 0.0821}
  ],
  "visualization": {
    "layer1": {"shape": [64, 32, 32], "values": [[...]]},
    "layer2": {"shape": [128, 16, 16], "values": [[...]]}
  },
  "input_spectrogram": {
    "shape": [128, 44],
    "values": [[...]]
  },
  "waveform": {
    "values": [...],
    "sample_rate": 44100,
    "duration": 5.0
  }
}
```

## 🏋️ Model Architecture

### AudioCNN Architecture

- **Base Architecture**: Custom ResNet-inspired CNN with residual connections
- **Input Processing**: Mel-spectrograms (128 mel bins, 1024 FFT, 512 hop length)
- **Sampling Rate**: 22050 Hz with frequency range 0-11025 Hz
- **Residual Blocks**: 4 progressive layers with channel expansion (64→128→256→512)
- **Feature Maps**: Intermediate layer visualization and extraction capability
- **Output**: 50-class softmax classification for ESC-50 categories
- **Regularization**: Dropout (0.5), Batch Normalization, Label Smoothing (0.1)

### ResidualBlock Components

- **Convolutional Layers**: 3x3 kernels with stride control
- **Batch Normalization**: Applied after each convolution
- **Shortcut Connections**: Identity mapping with 1x1 convolution when needed
- **Activation**: ReLU activation functions
- **Adaptive Pooling**: Global average pooling for final feature aggregation

## 📈 Training Configuration

### Dataset & Augmentation

- **Dataset**: ESC-50 (2000 samples, 50 environmental sound classes), [ESC-50 GitHub](https://github.com/karolpiczak/ESC-50)
- **Data Split**: Fold 1-4 for training, Fold 5 for validation
- **Audio Format**: 44.1kHz sampling rate, mono channel conversion
- **Augmentation**: Mixup (30% probability) + SpecAugment (frequency/time masking)

### Training Hyperparameters

- **Batch Size**: 32 samples per batch
- **Epochs**: 100 training epochs
- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate**: OneCycleLR scheduler (max_lr=0.002)
- **Loss Function**: CrossEntropyLoss with label smoothing (0.1)
- **Hardware**: Modal A10G GPU with automatic scaling

### Monitoring & Logging

- **TensorBoard**: Real-time loss, accuracy, and learning rate tracking
- **Model Checkpointing**: Best validation accuracy model preservation
- **Volume Persistence**: Automatic model and data storage on Modal volumes

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. **Modal Authentication Problems**

```powershell
# Re-authenticate with Modal
modal setup
```

#### 2. **GPU Memory Issues**

- Reduce batch size in `train.py` (line ~200)
- Use CPU inference by modifying device selection
- Clear CUDA cache: `torch.cuda.empty_cache()`

#### 3. **Audio Format Compatibility**

```powershell
# Install additional audio codecs
pip install librosa soundfile
```

- Ensure audio files are in supported formats (WAV, MP3, FLAC)
- Check sample rate (automatically resampled to 44.1kHz)
- Verify file integrity and format

#### 4. **Modal Deployment Issues**

```powershell
# Check Modal app status
modal app list

# Redeploy if needed
modal serve main.py --reload
```

#### 5. **Frontend Connection Issues**

- Verify Modal inference endpoint is running
- Update API endpoint URL in frontend configuration
- Check browser console for CORS or network errors
- Ensure Modal endpoint allows public access

#### 6. **Package Installation Problems**

```powershell
# Clear pip cache and reinstall
pip cache purge
pip install -r requirements.txt --force-reinstall

# For Node.js issues
cd audio-cnn-visualizer
rm -rf node_modules package-lock.json
npm install
```

#### 7. **TensorBoard Access Issues**

- Training logs are stored in Modal volumes
- Use Modal dashboard to view training progress
- Download logs locally for detailed analysis

### Performance Optimization

- **Inference Speed**: Use Modal's autoscaling for production loads
- **Memory Usage**: Implement batch processing for multiple files
- **Model Size**: Consider model quantization for edge deployment

## � References & Resources

- **[ESC-50 Dataset](https://github.com/karolpiczak/ESC-50)** - Environmental Sound Classification dataset
- **[Modal Documentation](https://modal.com/docs)** - Serverless cloud computing platform
- **[PyTorch Audio](https://pytorch.org/audio/)** - Audio processing with PyTorch
- **[ResNet Paper](https://arxiv.org/abs/1512.03385)** - Deep Residual Learning for Image Recognition
- **[Next.js Documentation](https://nextjs.org/docs)** - React framework for production
- **[Tailwind CSS](https://tailwindcss.com/)** - Utility-first CSS framework

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for frontend development
- Add type hints and docstrings
- Test your changes thoroughly
- Update documentation as needed
