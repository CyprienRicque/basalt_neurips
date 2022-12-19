import json
import os
import tkinter as tk
from tkinter import ttk
from typing import Optional

from PIL import Image, ImageTk
import cv2

from labeling_gui.jsonl_manager import JSONLManager

TIMELINE_HEIGHT = 80
FOLDER = "../basalt_neurips_data/BuildWaterFall/"

def count_lines_jsonl(file) -> int:
    with open(file, 'r') as f:
        count = 0

        for line in f:
            _ = json.loads(line)
            count += 1
    return count


def find_another_video(folder) -> Optional[str]:
    for file in os.listdir(folder):
        if file.endswith('.mp4') and file.split('.')[0] + '_annotations.jsonl' not in os.listdir(folder):
            return file
    return None


class MainWindow:
    def __init__(self, root,
                 file=FOLDER + 'cheeky-cornflower-setter-01b256a7cfb9-20220717-131452.mp4'):
        root.geometry("1500x900")

        self.load_video(file)

        width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

        self.interval_between_frames = int(1000 / self.video_capture.get(cv2.CAP_PROP_FPS))

        # Create a video_frame frame to hold the video and buttons
        video_frame = tk.Frame(root)
        video_frame.pack(side=tk.LEFT, fill=tk.NONE, expand=True)

        # Create a frame to hold the preview and timeline
        preview_frame = tk.Frame(root)
        preview_frame.pack(side=tk.RIGHT, fill=tk.NONE, expand=True)

        # Create a canvas to display the main video
        self.canvas = tk.Canvas(video_frame, bg='black', width=width, height=height)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a label to display the preview image
        self.preview_label = tk.Label(preview_frame, bg='black')
        self.preview_label.pack(side=tk.TOP, fill=tk.NONE, expand=False)

        # Create a canvas to display the timeline
        self.timeline_canvas = tk.Canvas(preview_frame, height=TIMELINE_HEIGHT, bg='gray')
        self.timeline_canvas.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

        # Bind events to the timeline canvas
        self.timeline_canvas.bind('<Button-1>', self.on_timeline_click)
        self.timeline_canvas.bind('<Motion>', self.on_timeline_motion)

        # Create a frame to hold the buttons
        buttons_frame = tk.Frame(video_frame)
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

        # Create the "Play/Pause" button
        self.play_pause_button = tk.Button(buttons_frame, text='Play', command=self.on_play_pause_button_click)
        self.play_pause_button.pack(side=tk.LEFT)

        self.bind_buttons(buttons_frame)
        self.bind_keys(root)

        # Set up the preview canvas
        self.update_preview(0)

        root.update_idletasks()

        # Set up the timeline canvas
        self.update_timeline()

        # Start the video loop
        self.video_loop()

    def bind_keys(self, root):
        # Binds
        root.bind("<Right>", self.on_skip_forward_button_click)
        root.bind("<Left>", self.on_skip_backward_button_click)
        root.bind("<Control-Right>", self.on_one_frame_skip_forward_button)
        root.bind("<Control-Left>", self.on_one_frame_skip_backward_button)
        root.bind("<r>", self.on_restart_button_click)
        root.bind("<f>", self.on_speed_up_button_click)
        root.bind("<s>", self.on_slow_down_button_click)
        root.bind("<p>", self.on_play_pause_button_click)
        root.bind("<space>", self.on_play_pause_button_click)
        root.bind("<n>", self.open_next_video)

    def bind_buttons(self, buttons_frame):
        # Create the "Speed up" button
        tk.Button(buttons_frame, text='>>', command=self.on_speed_up_button_click).pack(side=tk.LEFT)
        # Create the "Slow down" button
        tk.Button(buttons_frame, text='<<', command=self.on_slow_down_button_click).pack(side=tk.LEFT)
        # Create the "Rewind" button
        tk.Button(buttons_frame, text='Restart', command=self.on_restart_button_click).pack(side=tk.LEFT)
        # Create the "Skip Forward" button
        tk.Button(buttons_frame, text='Skip Forward', command=self.on_skip_forward_button_click).pack(side=tk.LEFT)
        # Create the "Skip Backward" button
        tk.Button(buttons_frame, text='Skip Backward', command=self.on_skip_backward_button_click).pack(side=tk.LEFT)

    def on_play_pause_button_click(self, event=None):
        # Toggle the play/pause state of the video
        if self.play_pause_button['text'] == 'Play':
            self.play_pause_button['text'] = 'Pause'
        else:
            self.play_pause_button['text'] = 'Play'

    SHIFT = 4

    def on_speed_up_button_click(self, event=None):
        if self.interval_between_frames - MainWindow.SHIFT > 0:
            self.interval_between_frames -= MainWindow.SHIFT

    def on_slow_down_button_click(self, event=None):
        self.interval_between_frames += MainWindow.SHIFT

    def on_restart_button_click(self, event=None):
        # Restart the video by setting the current frame to the start of the video
        self.set_main_timestamp(0)
        self._update_main_video()
        self.update_timeline()

    def on_skip_forward_button_click(self, event=None):
        # Skip forward 5 seconds in the video
        self.set_main_timestamp(self.current_timestamp + 5)
        self._update_main_video()
        self.update_timeline()

    def on_skip_backward_button_click(self, event=None):
        # Skip backward 5 seconds in the video
        self.set_main_timestamp(self.current_timestamp - 5)
        self._update_main_video()
        self.update_timeline()

    def on_one_frame_skip_forward_button(self, event=None):
        # Skip forward 5 seconds in the video
        self.set_main_frame(self.current_frame + 1)
        self._update_main_video()
        self.update_timeline()

    def on_one_frame_skip_backward_button(self, event=None):
        # Skip backward 5 seconds in the video
        self.set_main_frame(self.current_frame - 1)
        self._update_main_video()
        self.update_timeline()

    def on_timeline_click(self, event=None):
        # Calculate the timestamp of the clicked location on the timeline
        clicked_timestamp = event.x / self.timeline_canvas.winfo_width() * self.total_duration

        # Set the current frame to the corresponding frame in the video
        self.set_main_timestamp(clicked_timestamp)
        self._update_main_video()
        self.update_timeline()

    def open_next_video(self, event=None):
        self.jsonl_manager.dump()
        file = find_another_video(FOLDER)
        self.load_video(FOLDER + file)

    def load_video(self, file):
        # Open the video file
        self.video_capture = cv2.VideoCapture(file)
        self.video_capture_preview = cv2.VideoCapture(file)

        # Create a variable to store the current frame of the video
        self.current_frame, self.current_timestamp = 0, 0

        # Create a variable to store the total duration of the video
        self.total_frames_video = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.number_of_lines_jsonl = count_lines_jsonl(file.replace('mp4', 'jsonl'))

        # The first frame of the video is the word loading and therefore does not count
        assert self.total_frames_video - 1 == self.number_of_lines_jsonl, f"{self.total_frames_video=} {self.number_of_lines_jsonl=}"

        self.total_duration = self.total_frames_video / self.video_capture.get(cv2.CAP_PROP_FPS)

        self.jsonl_manager = JSONLManager(file.replace('.mp4', '_annotations.jsonl'), self.number_of_lines_jsonl)

    def on_timeline_motion(self, event=None):
        # Calculate the timestamp of the current mouse position on the timeline
        current_timestamp = event.x / self.timeline_canvas.winfo_width() * self.total_duration

        # Update the preview image to show the frame at the current timestamp
        self.update_preview(current_timestamp)

    def update_preview(self, timestamp):
        # Set the video capture to the frame at the specified timestamp
        self.video_capture_preview.set(cv2.CAP_PROP_POS_FRAMES,
                                       timestamp * self.video_capture_preview.get(cv2.CAP_PROP_FPS))

        # Read the frame at the specified timestamp
        _, frame = self.video_capture_preview.read()

        # Convert the frame to a PhotoImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame)
        frame_photo_image = ImageTk.PhotoImage(frame_image)

        # Update the preview label with the new frame
        self.preview_label.configure(image=frame_photo_image)
        self.preview_label.image = frame_photo_image

    def update_timeline(self):
        # Clear the timeline canvas
        self.timeline_canvas.delete('all')

        # Calculate the width of the timeline
        timeline_width = self.timeline_canvas.winfo_width()
        print(f"{timeline_width=}")
        # Add margin
        xy_margin = 20
        timeline_width -= 2 * xy_margin

        # Draw the current timestamp indicator on the timeline
        current_timestamp_x = (self.current_timestamp / self.total_duration * timeline_width) + xy_margin
        print(f"line at {current_timestamp_x}")
        self.timeline_canvas.create_line(current_timestamp_x, 20, current_timestamp_x, TIMELINE_HEIGHT, fill='red')
        self.timeline_canvas.create_text(current_timestamp_x, 10,
                                         text=f"{int(-1 if self.current_frame is None else self.current_frame)}",
                                         fill='black')

        ticks_count = 5

        for i in range(0, ticks_count + 1):
            # Draw the timeline ticks
            tick_x = i * timeline_width / ticks_count + xy_margin
            self.timeline_canvas.create_line(tick_x, TIMELINE_HEIGHT - 20, tick_x, TIMELINE_HEIGHT - 12, fill='white')

            # Draw the timeline labels
            secs = i * self.total_duration / ticks_count
            label_text = f"{int(secs // 60):02d}:{int(secs % 60):02d}"
            self.timeline_canvas.create_text(tick_x, TIMELINE_HEIGHT - 11, text=label_text, anchor='n', fill='white')

    def update_current_timestamp_from_frame(self):
        self.current_timestamp = self.current_frame / self.video_capture.get(cv2.CAP_PROP_FPS)

    def video_loop(self):
        # Check if the video is paused
        if self.is_playing():
            # Update the current frame and timestamp
            self.current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            self.update_current_timestamp_from_frame()

            self._update_main_video()

            # Update the timeline
            self.update_timeline()

        # Call the video loop again after a delay
        self.canvas.after(self.interval_between_frames, self.video_loop)

    def is_playing(self):
        return self.play_pause_button['text'] == 'Pause'

    def set_main_timestamp(self, timestamp):
        if timestamp < 0:
            timestamp = 0
        self.current_timestamp = timestamp
        self.current_frame = self.current_timestamp * self.video_capture_preview.get(cv2.CAP_PROP_FPS)
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def set_main_frame(self, frame):
        if frame < 0:
            frame = 0
        self.current_frame = frame
        self.update_current_timestamp_from_frame()
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def _update_main_video(self):
        # Read the next frame from the video
        _, frame = self.video_capture.read()

        # Check if the end of the video has been reached
        if frame is None:
            # Reset the video to the start
            self.set_main_timestamp(0)
            _, frame = self.video_capture.read()

        # Convert the frame to a PhotoImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame)
        frame_photo_image = ImageTk.PhotoImage(frame_image)

        # Update the canvas with the new frame
        self.canvas.create_image(0, 0, image=frame_photo_image, anchor='nw')
        self.canvas.image = frame_photo_image


# Create the main window
if __name__ == "__main__":
    root = tk.Tk()
    main_window = MainWindow(root)
    root.mainloop()
