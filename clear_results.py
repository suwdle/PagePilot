
import os
import shutil
import config

def clear_screenshot_directory():
    """Deletes all files in the screenshot directory specified in config."""
    dir_path = config.SCREENSHOT_DIR
    
    if not os.path.isdir(dir_path):
        print(f"Directory not found, nothing to clear: {dir_path}")
        return

    # List all files in the directory
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    
    print(f"Successfully cleared all files in {dir_path}")

if __name__ == "__main__":
    clear_screenshot_directory()
