# Filter warnings before any imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weight_norm.*")
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")

import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from dia.model import Dia

# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument(
    "--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')"
)
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")

args = parser.parse_args()


# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load Nari model and config
print("Loading Nari model...")
try:
    # Use the function from inference.py
    model = Dia.from_pretrained("nari-labs/Dia-1.6B")
except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise

def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
):
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Uses temporary files for text and audio prompt compatibility with inference.generate.
    """
    # global model, device  # Access global model, config, device

    if not text_input or text_input.isspace():
        raise gr.Error("Text input cannot be empty.")

    temp_txt_file_path = None
    temp_audio_prompt_path = None
    output_audio = (44100, np.zeros(1, dtype=np.float32))

    try:
        prompt_path_for_generate = None
        if audio_prompt_input is not None:
            sr, audio_data = audio_prompt_input
            # Check if audio_data is valid
            if (
                audio_data is None or audio_data.size == 0 or audio_data.max() == 0
            ):  # Check for silence/empty
                gr.Warning("Audio prompt seems empty or silent, ignoring prompt.")
            else:
                # Save prompt audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".wav", delete=False
                ) as f_audio:
                    temp_audio_prompt_path = f_audio.name  # Store path for cleanup

                    # Basic audio preprocessing for consistency
                    # Convert to float32 in [-1, 1] range if integer type
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        gr.Warning(
                            f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion."
                        )
                        # Attempt conversion, might fail for complex types
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(
                                f"Failed to convert audio prompt to float32: {conv_e}"
                            )

                    # Ensure mono (average channels if stereo)
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:  # Assume (2, N)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:  # Assume (N, 2)
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(
                                f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis."
                            )
                            audio_data = (
                                audio_data[0]
                                if audio_data.shape[0] < audio_data.shape[1]
                                else audio_data[:, 0]
                            )
                        audio_data = np.ascontiguousarray(
                            audio_data
                        )  # Ensure contiguous after slicing/mean

                    # Write using soundfile
                    try:
                        sf.write(
                            temp_audio_prompt_path, audio_data, sr, subtype="FLOAT"
                        )  # Explicitly use FLOAT subtype
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(
                            f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})"
                        )
                    except Exception as write_e:
                        print(f"Error writing temporary audio file: {write_e}")
                        raise gr.Error(f"Failed to save audio prompt: {write_e}")

        # 3. Run Generation

        start_time = time.time()

        # Use torch.inference_mode() context manager for the generation call
        with torch.inference_mode():
            output_audio_np = model.generate(
                text_input,
                max_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                use_cfg_filter=True,
                cfg_filter_top_k=cfg_filter_top_k,  # Pass the value here
                use_torch_compile=False,  # Keep False for Gradio stability
                audio_prompt_path=prompt_path_for_generate,
            )

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        # 4. Convert Codes to Audio
        if output_audio_np is not None:
            # Get sample rate from the loaded DAC model
            output_sr = 44100

            # --- Slow down audio ---
            original_len = len(output_audio_np)
            # Ensure speed_factor is positive and not excessively small/large to avoid issues
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(
                original_len / speed_factor
            )  # Target length based on speed_factor
            if (
                target_len != original_len and target_len > 0
            ):  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio = (
                    output_sr,
                    resampled_audio_np.astype(np.float32),
                )  # Use resampled audio
                print(
                    f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed."
                )
            else:
                output_audio = (
                    output_sr,
                    output_audio_np,
                )  # Keep original if calculation fails or no change
                print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")
            # --- End slowdown ---

            print(
                f"Audio conversion successful. Final shape: {output_audio[1].shape}, Sample Rate: {output_sr}"
            )

        else:
            print("\nGeneration finished, but no valid tokens were produced.")
            # Return default silence
            gr.Warning("Generation produced no output.")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()
        # Re-raise as Gradio error to display nicely in the UI
        raise gr.Error(f"Inference failed: {e}")

    finally:
        # 5. Cleanup Temporary Files defensively
        if temp_txt_file_path and Path(temp_txt_file_path).exists():
            try:
                Path(temp_txt_file_path).unlink()
                print(f"Deleted temporary text file: {temp_txt_file_path}")
            except OSError as e:
                print(
                    f"Warning: Error deleting temporary text file {temp_txt_file_path}: {e}"
                )
        if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
            try:
                Path(temp_audio_prompt_path).unlink()
                print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
            except OSError as e:
                print(
                    f"Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}"
                )

    return output_audio


# --- Create Gradio Interface ---
css = """
:root {
    --primary-color: #ff6b35;
    --secondary-color: #2e2e2e;
    --background-color: #121212;
    --text-color: #f5f5f5;
    --panel-bg: #1e1e1e;
    --border-radius: 8px;
    --shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}

body {
    background-color: var(--background-color);
    color: var(--text-color);
    margin: 0;
    font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

#main-container {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 20px;
}

#header {
    margin-bottom: 30px;
    text-align: center;
    border-bottom: 1px solid #333;
    padding-bottom: 15px;
    position: relative;
}

.fancy-title {
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 10px !important;
    text-shadow: 0 0 3px rgba(255, 107, 53, 0.15);
    position: relative;
    padding: 10px 0;
}

.fancy-title:after {
    content: '';
    position: absolute;
    width: 150px;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-color), #ff9966);
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    border-radius: 2px;
}

.fancy-title span {
    background: linear-gradient(90deg, var(--primary-color), #ff9966);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    display: inline-block;
    position: relative;
}

.fancy-title span:before {
    content: "Nari Text-to-Speech Synthesis";
    position: absolute;
    left: -0.3px;
    top: -0.3px;
    background: linear-gradient(90deg, #ffaa73, #ffcc99);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    z-index: -1;
    filter: blur(0.1px);
}

.fancy-subtitle {
    font-style: italic;
    color: #aaa;
    font-size: 1.1rem;
    font-weight: 300;
    margin-top: 5px;
}

.input-panel, .output-panel {
    background-color: var(--panel-bg);
    border-radius: var(--border-radius);
    padding: 15px;
    box-shadow: var(--shadow);
    transition: all 0.3s ease;
}

.input-panel:hover, .output-panel:hover {
    box-shadow: 0 6px 10px rgba(0, 0, 0, 0.3);
}

button.primary {
    background: linear-gradient(90deg, var(--primary-color), #ff9966) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: var(--border-radius) !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
    box-shadow: var(--shadow) !important;
}

button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3) !important;
}

.waveform-display {
    border-radius: var(--border-radius);
    overflow: hidden;
    background-color: rgba(30, 30, 30, 0.7);
}

.audio-controls {
    margin-top: 10px;
    display: flex;
    justify-content: center;
}

.accordion-header {
    font-weight: bold;
    color: var(--primary-color);
}

textarea, input[type="text"] {
    background-color: rgba(45, 45, 45, 0.5) !important;
    color: var(--text-color) !important;
    border: 1px solid #444 !important;
    border-radius: var(--border-radius) !important;
}

.footer {
    margin-top: 20px;
    text-align: center;
    font-size: 0.8rem;
    color: #666;
}

.slider-container .label {
    color: #bbb !important;
}

.gradio-slider input[type="range"] {
    accent-color: var(--primary-color) !important;
}
"""

# Attempt to load default text from example.txt
default_text = "[S1] Dia is an open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] Wow. Amazing. (laughs) \n[S2] Try it now on Git hub or Hugging Face."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:  # Handle empty example file
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")


# Build Gradio UI
with gr.Blocks(css=css, theme=gr.themes.Monochrome(), title="Nari-Dia") as demo:
    with gr.Column(elem_id="main-container"):
        with gr.Row(elem_id="header"):
            gr.HTML("""
                <div class="fancy-title">
                    <span>Nari Text-to-Speech Synthesis</span>
                </div>
                <div class="fancy-subtitle">
                    Transform your scripts into natural dialogue with voice control
                </div>
            """)
        
        with gr.Row():
            with gr.Column(elem_classes="input-panel"):
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text here...",
                    value=default_text,
                    lines=6,
                )
                
                with gr.Row():
                    audio_prompt_input = gr.Audio(
                        label="Audio Prompt (Optional)",
                        show_label=True,
                        sources=["upload", "microphone"],
                        type="numpy",
                        elem_classes="waveform-display"
                    )
                
                with gr.Accordion("Generation Parameters", open=False, elem_classes="accordion-header"):
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            max_new_tokens = gr.Slider(
                                label="Max New Tokens (Audio Length)",
                                minimum=860,
                                maximum=3072,
                                value=model.config.data.audio_length,
                                step=50,
                                info="Controls the maximum length of generated audio",
                                elem_classes="slider-container"
                            )
                            
                            cfg_scale = gr.Slider(
                                label="CFG Scale",
                                minimum=1.0,
                                maximum=5.0,
                                value=3.0,
                                step=0.1,
                                info="Higher values = stronger text adherence",
                                elem_classes="slider-container"
                            )
                            
                            temperature = gr.Slider(
                                label="Temperature",
                                minimum=1.0,
                                maximum=1.5,
                                value=1.3,
                                step=0.05,
                                info="Randomness factor",
                                elem_classes="slider-container"
                            )
                            
                        with gr.Column():
                            top_p = gr.Slider(
                                label="Top P",
                                minimum=0.80,
                                maximum=1.0,
                                value=0.95,
                                step=0.01,
                                info="Nucleus sampling probability",
                                elem_classes="slider-container"
                            )
                            
                            cfg_filter_top_k = gr.Slider(
                                label="CFG Filter Top K",
                                minimum=15,
                                maximum=50,
                                value=30,
                                step=1,
                                info="Top k filter for CFG guidance",
                                elem_classes="slider-container"
                            )
                            
                            speed_factor_slider = gr.Slider(
                                label="Speed Factor",
                                minimum=0.8,
                                maximum=1.0,
                                value=0.94,
                                step=0.02,
                                info="Adjust speech speed",
                                elem_classes="slider-container"
                            )
                
                run_button = gr.Button("Generate Audio", variant="primary", elem_classes="primary")

            with gr.Column(elem_classes="output-panel"):
                gr.Markdown("### Generated Audio")
                audio_output = gr.Audio(
                    label="",
                    type="numpy",
                    autoplay=False,
                    elem_classes="waveform-display"
                )
                
        with gr.Row(elem_classes="footer"):
            gr.Markdown("Built with Gradio • [Use via API](https://huggingface.co/nari-labs/Dia-1.6B) • [Settings](#)")

    # Link button click to function
    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
        ],
        outputs=[audio_output],
        api_name="generate_audio",
    )


# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=args.share)
