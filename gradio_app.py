import gradio as gr
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime

CONFIG_PATH = Path("configs/unet/stage2.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")


def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    face_upscale_factor,
    face_enhance,
    face_enhance_method,
    face_enhance_strength,
    high_quality,
    seed,
):
    try:
        # Create the temp directory if it doesn't exist
        output_dir = Path("./results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique subfolder for this run
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{current_time}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Convert paths to absolute Path objects and normalize them
        video_file_path = Path(video_path)
        video_path = video_file_path.absolute().as_posix()
        audio_path = Path(audio_path).absolute().as_posix()

        # Set the output path for the processed video
        output_path = str(run_dir / f"{video_file_path.stem}_processed.mp4")  # Change the filename as needed

        config = OmegaConf.load(CONFIG_PATH)

        config["run"].update(
            {
                "guidance_scale": guidance_scale,
                "inference_steps": inference_steps,
            }
        )

        # Parse the arguments
        args = create_args(
            video_path, 
            audio_path, 
            output_path, 
            inference_steps, 
            guidance_scale, 
            face_upscale_factor,
            face_enhance,
            face_enhance_method,
            face_enhance_strength,
            high_quality,
            seed
        )

        print(f"Processing with face_upscale_factor={face_upscale_factor} and high_quality={high_quality}")
        print(f"Face enhance: {face_enhance}, method: {face_enhance_method}, strength: {face_enhance_strength}")
        print(f"Input video: {video_path}")
        print(f"Input audio: {audio_path}")
        print(f"Output path: {output_path}")

        result = main(
            config=config,
            args=args,
        )
        print("Processing completed successfully.")
        return output_path  # Ensure the output path is returned
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during processing: {str(e)}\n{error_details}")
        raise gr.Error(f"处理时出错: {str(e)}")


def create_args(
    video_path: str, 
    audio_path: str, 
    output_path: str, 
    inference_steps: int, 
    guidance_scale: float,
    face_upscale_factor: float,
    face_enhance: bool,
    face_enhance_method: str,
    face_enhance_strength: float,
    high_quality: bool,
    seed: int
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--face_upscale_factor", type=float, default=1.0)
    parser.add_argument("--face_enhance", action="store_true")
    parser.add_argument("--face_enhance_method", type=str, default="combined")
    parser.add_argument("--face_enhance_strength", type=float, default=0.8)
    parser.add_argument("--high_quality", action="store_true")
    parser.add_argument("--seed", type=int, default=1247)

    args_list = [
        "--inference_ckpt_path",
        CHECKPOINT_PATH.absolute().as_posix(),
        "--video_path",
        video_path,
        "--audio_path",
        audio_path,
        "--video_out_path",
        output_path,
        "--inference_steps",
        str(inference_steps),
        "--guidance_scale",
        str(guidance_scale),
        "--face_upscale_factor",
        str(face_upscale_factor),
        "--face_enhance_method",
        face_enhance_method,
        "--face_enhance_strength",
        str(face_enhance_strength),
        "--seed",
        str(seed),
    ]
    
    # 布尔标志选项
    if high_quality:
        args_list.append("--high_quality")
    if face_enhance:
        args_list.append("--face_enhance")
    
    return parser.parse_args(args_list)


# Create Gradio interface
with gr.Blocks(title="LatentSync Video Processing") as demo:
    gr.Markdown(
        """
    # LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision
    Upload a video and audio file to process with LatentSync model.

    <div align="center">
        <strong>Chunyu Li1,2  Chao Zhang1  Weikai Xu1  Jinghui Xie1,†  Weiguo Feng1
        Bingyue Peng1  Weiwei Xing2,†</strong>
    </div>

    <div align="center">
        <strong>1ByteDance   2Beijing Jiaotong University</strong>
    </div>

    <div style="display:flex;justify-content:center;column-gap:4px;">
        <a href="https://github.com/bytedance/LatentSync">
            <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
        </a> 
        <a href="https://arxiv.org/pdf/2412.09262">
            <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
        </a>
    </div>
    """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input Video")
            audio_input = gr.Audio(label="Input Audio", type="filepath")

            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=2.5,
                    value=1.5,
                    step=0.5,
                    label="Guidance Scale",
                )
                inference_steps = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="Inference Steps")
            
            with gr.Row():
                face_upscale_factor = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Face Upscale Factor",
                    info="Higher values improve face details (1.0-2.0)"
                )
                face_enhance = gr.Checkbox(
                    label="Face Enhance", 
                    value=False,
                    info="Enable for face enhancement"
                )

            with gr.Row():
                face_enhance_method = gr.Dropdown(
                    choices=["gpen", "gfpgan", "codeformer"],
                    value="gfpgan",
                    label="Face Enhance Method",
                    info="Select the method for face enhancement"
                )
                face_enhance_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.1,
                    label="Face Enhance Strength",
                    info="Adjust the strength of face enhancement"
                )

            with gr.Row():
                high_quality = gr.Checkbox(
                    label="High Quality Output", 
                    value=False,
                    info="Enable for better video quality (slower)"
                )

            with gr.Row():
                seed = gr.Number(value=1247, label="Random Seed", precision=0)

            process_btn = gr.Button("Process Video")

        with gr.Column():
            video_output = gr.Video(label="Output Video")

            gr.Examples(
                examples=[
                    ["assets/demo1_video.mp4", "assets/demo1_audio.wav"],
                    ["assets/demo2_video.mp4", "assets/demo2_audio.wav"],
                    ["assets/demo3_video.mp4", "assets/demo3_audio.wav"],
                ],
                inputs=[video_input, audio_input],
            )

    process_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            audio_input,
            guidance_scale,
            inference_steps,
            face_upscale_factor,
            face_enhance,
            face_enhance_method,
            face_enhance_strength,
            high_quality,
            seed,
        ],
        outputs=video_output,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", type=bool, default=False)
    args = parser.parse_args()
    
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True
    )