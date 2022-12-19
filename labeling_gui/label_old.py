import tkinter as tk
import tkinter.ttk as ttk
import cv2

from PIL import Image, ImageTk

# Note: You will need to define the functions `go_back`, `advance`, `play_pause`, `fast_forward`, `set_timestamp_on_click`, and `update_preview_on_mouse_move` to implement the desired functionality.
# Here's some sample code for these functions:

def go_back(event):
    global current_timestamp
    current_timestamp -= 5
    if current_timestamp < 0:
        current_timestamp = 0
    set_timestamp(current_timestamp)

def advance(event):
    global current_timestamp
    current_timestamp += 5
    if current_timestamp > duration:
        current_timestamp = duration
    set_timestamp(current_timestamp)

def play_pause(event):
    global playing
    if playing:
        playing = False
    else:
        playing = True

def fast_forward(event):
    global current_timestamp
    current_timestamp += 30
    if current_timestamp > duration:
        current_timestamp = duration
    set_timestamp(current_timestamp)

def set_timestamp_on_click(event):
    global current_timestamp
    current_timestamp = int(timeline_scale.get())
    set_timestamp(current_timestamp)


def update_preview_on_mouse_move(event):
    global current_timestamp
    current_timestamp = int(timeline_scale.get())
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_timestamp)
    _, frame = video_capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image = ImageTk.PhotoImage(image)
    preview_var.set(image)


def video_loop():
    global current_timestamp
    global playing

    # If the video is playing, update the current timestamp and display the next frame
    if playing:
        current_timestamp += 1
        if current_timestamp > duration:
            current_timestamp = 0
        set_timestamp(current_timestamp)

    # Read the next frame from the video capture
    _, frame = video_capture.read()

    # If the frame is not None, display it on the video canvas
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image)
        video_canvas.create_image(0, 0, anchor=tk.NW, image=image)
        video_canvas.image = image

    # Call the video loop again after a delay
    root.after(int(1000 / video_capture.get(cv2.CAP_PROP_FPS)), video_loop)


def set_timestamp(timestamp):
    global current_timestamp
    current_timestamp = timestamp
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_timestamp)
    timeline_scale.set(current_timestamp)


# Initialize the playing variable as False
playing = False

# Create the main window
root = tk.Tk()
root.title("Video Player")

# Create the video frame
video_frame = tk.Frame(root)
video_frame.pack(side=tk.TOP)

# Create the video canvas and bind it to the video frame
video_canvas = tk.Canvas(video_frame, width=640, height=480)
video_canvas.pack()

# Create the timeline frame
timeline_frame = tk.Frame(root)
timeline_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Create the timeline scale and bind it to the timeline frame
timeline_scale = ttk.Scale(timeline_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                            command=lambda x: set_timestamp(int(x)))
timeline_scale.pack(side=tk.BOTTOM, fill=tk.X)

# Create the preview frame and bind it to the video frame
preview_frame = tk.Frame(video_frame, width=320, height=240)
preview_frame.pack(side=tk.TOP)

# Create the preview canvas and bind it to the preview frame
preview_canvas = tk.Canvas(preview_frame, width=320, height=240)
preview_canvas.pack()

# Create the preview label and bind it to the preview canvas
preview_label = tk.Label(preview_canvas)
preview_label.pack()

# Create a StringVar to update the preview label
preview_var = tk.StringVar()
preview_label.config(textvariable=preview_var)

# Initialize the video capture and get the video's duration
video_capture = cv2.VideoCapture("../basalt_neurips_data/BuildWaterFall/woozy-ruby-ostrich-ff3ddf992ab4-20220721-125851.mp4")
duration = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Set the timeline scale's maximum value to the video's duration
timeline_scale.config(to=duration)

# Set the video and preview frames to their initial positions
set_timestamp(0)

# Start the video loop
root.after(0, video_loop)

# Bind the left arrow key to the go_back function
root.bind("<Left>", go_back)

# Bind the right arrow key to the advance function
root.bind("<Right>", advance)

# Bind the p key to the play_pause function
root.bind("<p>", play_pause)

# Bind the f key to the fast_forward function
root.bind("<f>", fast_forward)

# Bind the timeline scale to the set_timestamp_on_click function
timeline_scale.bind("<Button-1>", set_timestamp_on_click)

# Bind the timeline scale to the update_preview_on_mouse_move function
timeline_scale.bind("<Motion>", update_preview_on_mouse_move)

# Run the tkinter event loop
root.mainloop()


