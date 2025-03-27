import pyautogui
import time

# Set the interval in seconds
interval = 5  # Change this to any interval you want

try:
    while True:
        pyautogui.click()  # Perform a single left-click
        print("Clicked!")
        time.sleep(interval)  # Wait for the given interval before clicking again
except KeyboardInterrupt:
    print("Script stopped by user.")

