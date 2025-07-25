import os
import shutil

def sync_folders(parent_dir, child_dir):
    """
    Deletes files from the child directory if they are not present in the parent directory.

    Args:
        parent_dir (str): The path to the source/parent folder.
        child_dir (str): The path to the target/child folder to be cleaned.
    """
    # --- 1. Validate that both paths are actual directories ---
    if not os.path.isdir(parent_dir):
        print(f"❌ Error: Parent directory not found at '{parent_dir}'")
        return
    if not os.path.isdir(child_dir):
        print(f"❌ Error: Child directory not found at '{child_dir}'")
        return

    print(f"🔍 Parent Folder: {parent_dir}")
    print(f"🔍 Child Folder:  {child_dir}\n")

    # --- 2. Get the set of filenames from both directories ---
    # Using sets makes finding differences very efficient.
    parent_files = {f for f in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, f))}
    child_files = {f for f in os.listdir(child_dir) if os.path.isfile(os.path.join(child_dir, f))}

    # --- 3. Find files that are in the child folder but not in the parent ---
    files_to_delete = child_files - parent_files

    # --- 4. Delete the files if any are found ---
    if not files_to_delete:
        print("✅ Folders are already in sync. No files were deleted.")
        return

    print(f"🔥 Deleting {len(files_to_delete)} file(s) from the child folder:")
    for filename in files_to_delete:
        file_path = os.path.join(child_dir, filename)
        try:
            os.remove(file_path)
            print(f"  - Deleted: {filename}")
        except OSError as e:
            print(f"  - ❌ Error deleting {filename}: {e}")

    print("\n✅ Synchronization complete.")


if __name__ == "__main__":
    # --- IMPORTANT: SET YOUR FOLDER PATHS HERE ---
    # Use raw strings (r"...") or forward slashes ("/") for paths to avoid issues.
    # Example for Windows: r"C:\Users\YourUser\Desktop\Parent"
    # Example for macOS/Linux: "/home/user/pictures/parent"

    parent_folder_path = r"classified_cars_output/back-left/cropped_images"
    child_folder_path = r"classified_cars_output/back-left/saliency_maps"
    # ----------------------------------------------

    # --- SAFETY WARNING ---
    print("🚨 WARNING: This script will permanently delete files. 🚨")
    # Uncomment the line below to run the script after setting your paths.
    sync_folders(parent_folder_path, child_folder_path)