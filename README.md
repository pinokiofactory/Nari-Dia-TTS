# Nari Dia - Text-to-Speech Application
![Screenshot 2025-04-22 121753](https://github.com/user-attachments/assets/1fad7fef-8736-4bb5-a94f-993fa2416a8e)


## Overview

Nari Dia is a powerful text-to-speech (TTS) application based on the Dia-1.6B model from Nari Labs. This application allows you to convert text into natural-sounding speech with various customization options.

## Features

- High-quality text-to-speech conversion
- Voice customization through audio prompts
- Adjustable parameters (temperature, top-p, CFG scale)
- Speed control for generated audio
- Easy-to-use Gradio interface

## Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended) or CPU
- PyTorch
- Gradio
- Other dependencies listed in `requirements.txt`

## Installation  
Install via [Pinokio](https://pinokio.computer).

1. Clone the repository:
   ```bash
   git clone https://github.com/SUP3RMASS1VE/Nari-Dia-App.git
   cd Nari-Dia-App
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On Linux/Mac
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   # Windows
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   # Mac
   pip3 install torch torchvision torchaudio

   pip install -r requirements.txt
   ```

## Usage

Run the application:

```bash
python app.py
```

For additional options:

```bash
python app.py --help
```

Available arguments:
- `--device`: Force specific device (e.g., 'cuda', 'mps', 'cpu')
- `--share`: Enable Gradio sharing to access the interface over the internet

## How It Works

The application uses the Dia model architecture to generate audio codes from text input, which are then converted to waveforms using a DAC (Descript Audio Codec) model. The process involves:

1. Text encoding and processing
2. Optional audio prompt processing (for voice cloning)
3. Generating audio tokens using the Dia model
4. Converting tokens to audio using the DAC model
5. Post-processing audio (such as speed adjustment)

## Repository

The official repository is available at: [https://github.com/nari-labs/dia](https://github.com/nari-labs/dia)

## License

See the LICENSE file for details.

## Acknowledgements

- Nari Labs for the Dia-1.6B model
- Hugging Face for model hosting
- Gradio for the web interface framework
