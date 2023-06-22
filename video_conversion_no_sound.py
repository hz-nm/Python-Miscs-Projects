from moviepy.editor import VideoFileClip
import os


all_files = os.listdir('./')
print(all_files[3:])

path = "./"
output_path = os.path.join(path, "output")
i = 0
for filename in all_files[3:4]:
    videoclip = VideoFileClip(filename)
    new_clip = videoclip.without_audio()
    new_clip.write_videofile(f"output/output_{i}.mp4")
    i += 1
    print(f"Converted {i} file/s...")

