import random
import subprocess
import os
import shutil
import gradio as gr
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

def convert(segment_length, video, audio, progress=gr.Progress()):
    if segment_length is None:
        segment_length = 0

    temp_video_dir = os.path.join(current_dir, "temp/video")
    temp_audio_dir = os.path.join(current_dir, "temp/audio")
    results_dir = os.path.join(current_dir, "results")

    os.makedirs(temp_video_dir, exist_ok=True)
    os.makedirs(temp_audio_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Validate and extract file paths
    if not isinstance(video, str) or not os.path.isfile(video):
        raise ValueError(f"Invalid video file path: {video}")
    if not isinstance(audio, str) or not os.path.isfile(audio):
        raise ValueError(f"Invalid audio file path: {audio}")

    # Copy files to avoid file lock issues
    video_copy = os.path.join(temp_video_dir, "temp_video.mp4")
    audio_copy = os.path.join(temp_audio_dir, "temp_audio.mp3")

    try:
        shutil.copy(video, video_copy)
        shutil.copy(audio, audio_copy)
    except Exception as e:
        raise OSError(f"Error copying files: {e}")

    if segment_length != 0:
        video_segments = cut_video_segments(video_copy, segment_length, temp_video_dir)
        audio_segments = cut_audio_segments(audio_copy, segment_length, temp_audio_dir)
    else:
        video_segments = [video_copy]
        audio_segments = [audio_copy]

    processed_segments = []
    for i, (video_seg, audio_seg) in progress.tqdm(
        enumerate(zip(video_segments, audio_segments)), total=len(video_segments)
    ):
        processed_output = process_segment(video_seg, audio_seg, i, results_dir)
        processed_segments.append(processed_output)

    output_file = os.path.join(results_dir, f"output_{random.randint(0, 1000)}.mp4")
    concatenate_videos(processed_segments, output_file)

    # Remove temporary files
    cleanup_temp_files(video_segments + audio_segments)

    return output_file

def cleanup_temp_files(file_list):
    for file_path in file_list:
        if os.path.isfile(file_path):
            os.remove(file_path)

def cut_video_segments(video_file, segment_length, temp_directory):
    shutil.rmtree(temp_directory, ignore_errors=True)
    os.makedirs(temp_directory, exist_ok=True)
    segment_template = os.path.join(temp_directory, f"{random.randint(0, 1000)}_%03d.mp4")
    command = ["ffmpeg", "-i", video_file, "-c", "copy", "-f",
               "segment", "-segment_time", str(segment_length), segment_template]
    try:
        subprocess.run(command, check=True)
        return [os.path.join(temp_directory, f) for f in os.listdir(temp_directory)]
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error during inference: {e}\nCommand: {e.cmd}\nOutput: {e.output}")    

def cut_audio_segments(audio_file, segment_length, temp_directory):
    shutil.rmtree(temp_directory, ignore_errors=True)
    os.makedirs(temp_directory, exist_ok=True)
    segment_template = os.path.join(temp_directory, f"{random.randint(0, 1000)}_%03d.mp3")
    command = ["ffmpeg", "-i", audio_file, "-f", "segment",
               "-segment_time", str(segment_length), segment_template]
    try:
        subprocess.run(command, check=True)
        return [os.path.join(temp_directory, f) for f in os.listdir(temp_directory)]
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error during inference: {e}\nCommand: {e.cmd}\nOutput: {e.output}")    

def process_segment(video_seg, audio_seg, i, results_dir):
    output_file = os.path.join(results_dir, f"{random.randint(10, 100000)}_{i}.mp4")
    python_executable = sys.executable  # Get the path to the current Python interpreter
    command = [python_executable, "inference.py", "--face", video_seg, "--audio", audio_seg, "--outfile", output_file]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error during inference: {e}\nCommand: {e.cmd}\nOutput: {e.output}")
    return output_file

def concatenate_videos(video_segments, output_file):
    with open("segments.txt", "w") as file:
        for segment in video_segments:
            file.write(f"file '{segment}'\n")
    command = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", "segments.txt", "-c", "copy", output_file]
    subprocess.run(command, check=True)
    os.remove("segments.txt")

with gr.Blocks(
    title="Audio-based Lip Synchronization",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
) as demo:
    with gr.Row():
        gr.Markdown("# Audio-based Lip Synchronization")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                seg = gr.Number(label="Segment length (seconds), 0 for no segmentation")
            with gr.Row():
                with gr.Column():
                    v = gr.Video(label="Source Face")

                with gr.Column():
                    a = gr.Audio(type="filepath", label="Target Audio")

            with gr.Row():
                btn = gr.Button(value="Synthesize", variant="primary")
            with gr.Row():
                gr.Examples(
                    label="Face Examples",
                    examples=[
                        os.path.join(current_dir, "examples/face/1.mp4"),
                        os.path.join(current_dir, "examples/face/2.mp4"),
                        os.path.join(current_dir, "examples/face/3.mp4"),
                        os.path.join(current_dir, "examples/face/4.mp4"),
                        os.path.join(current_dir, "examples/face/5.mp4"),
                    ],
                    inputs=[v],
                )
            with gr.Row():
                gr.Examples(
                    label="Audio Examples",
                    examples=[
                        os.path.join(current_dir, "examples/audio/1.wav"),
                        os.path.join(current_dir, "examples/audio/2.wav"),
                    ],
                    inputs=[a],
                )

        with gr.Column():
            o = gr.Video(label="Output Video")

    btn.click(fn=convert, inputs=[seg, v, a], outputs=[o])

demo.queue().launch()
