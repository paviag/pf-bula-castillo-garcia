import subprocess

scripts = ["preProcessing.py", "huMoments.py", "modelSetUp.py", "model.py"]

for script in scripts:
    print(f"Executing {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"Error in {script}:\n{result.stderr}")
    
    print(f"Finished executing {script}\n")

print("All scripts executed successfully.")
