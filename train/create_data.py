import glob
import cv2
import os
from tqdm import tqdm


def load_images_paths(path):
    images = []
    for file in glob.glob(path):
        images.append(file)
    return images

def extract_frames_from_video(video_path, output_dir, frame_skip=1):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_name} (skipped {frame_skip-1} frames between saves)")
    return saved_count


if __name__ == "__main__":
    
    video_folder = "data/videos/*.mp4"
    output_folder = "images"
    frames_to_skip = 20

    # Para sumar los frames extraidos
    total_frames_extracted = 0

    images_paths = load_images_paths(video_folder)
    if not images_paths:
        print(f"No videos found in {video_folder}")
    else:
        print(f"Starting frame extraction from {len(images_paths)} videos...")
        for video_path in tqdm(images_paths):
            total_frames_extracted += extract_frames_from_video(video_path, output_folder, frames_to_skip)

        print("\n" + "="*40)
        print(f"  Total frames extracted: {total_frames_extracted}")
        print("="*40 + "\n")
