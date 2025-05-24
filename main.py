import subprocess
import os
import time

window_capture = "getWindow.py" # code for calibrating the window dimensions
feedback = "blip2.py" # code for describing the code
# critique code using ollama will be inputted here
image_dir = "captured_images" # all images that have been taken goes here

# don't start describing image until the getWindow.py code runs and takes the first photo
def wait_for_first_image():
    print("⏳ Waiting for images to appear in 'captured_images/'...")
    while True:
        if os.listdir(image_dir):
            print("✅ Image(s) found. Starting feedback system...")
            return
        time.sleep(1)

# subprocess to run the scripts at the same time
def run_script_background(script_name):
    return subprocess.Popen(["python", script_name])

def main():
    os.makedirs(image_dir, exist_ok=True) # make a directory for the saved images

    print("Launching canvas capture")
    capture_proc = run_script_background(window_capture) # get window

    wait_for_first_image() # wait for photo

    print("Launching analysis")
    feedback_proc = run_script_background(feedback) # get description

    # print("Launching live critique")
    # critique script here

    # multithread coding here
    try:
        capture_proc.wait()
        feedback_proc.wait()
    except KeyboardInterrupt:
        print("\nTerminating process")
        capture_proc.terminate()
        feedback_proc.terminate()

# run script
if __name__ == "__main__":
    main()
