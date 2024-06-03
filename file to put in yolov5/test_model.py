import torch
import cv2
import os
from tqdm import tqdm

# Set paths
weights_path = 'yolov5/best.pt'  # Path to the trained model weights
source_dir = 'Tests'  # Input video directory
output_dir = 'Tests_result'  # Output video directory

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the model
model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local')

# Specify the video files in the desired order
video_files = ['02.mp4']
model.conf = 0.1

for video_file in video_files:
    source_video = os.path.join(source_dir, video_file)
    output_video = os.path.join(output_dir, video_file)

    # Open the video file
    cap = cv2.VideoCapture(source_video)

    # Get video properties: width, height, and frames per second (fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))  # Use original width and height

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 3000

    # Use tqdm to display a progress bar
    with tqdm(total=total_frames, desc=f"Processing {video_file}", unit="frame") as pbar:
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Pass the frame to the model for inference
            results = model(frame)

            # Render the results on the frame
            result_frame = results.render()[0]

            # Write the rendered frame to the output video
            out.write(result_frame)

            # Update the progress bar
            pbar.update(1)

    # Release the video capture and writer
    cap.release()
    out.release()

    print(f"Processing of {video_file} complete. Result saved to {output_video}")

print("All video processing complete.")
