import os

# Directory where files will be created
directory = os.getcwd()  # Use the current working directory

# Create files
for i in range(1, 501):
    # Format the filename with leading zeros
    filename = f"artwork{i:03d}.txt"
    file_path = os.path.join(directory, filename)
    
    # Check if the file already exists
    if not os.path.exists(file_path):
        # Create and close the file to ensure it is created
        with open(file_path, 'w') as file:
            # Optionally write something to the file, e.g., "Placeholder text"
            file.write("Placeholder text")
        print(f"Created: {filename}")
    else:
        print(f"File already exists: {filename}")

print("File creation process completed.")
