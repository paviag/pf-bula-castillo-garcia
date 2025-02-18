import subprocess

scripts = [
    # "convertImgs.py"
    "preProcessing.py", 
    #"huMoments.py", 
    "modelSetUp.py", 
    "model.py",
]

for script in scripts:
    sc = "pf-bula-castillo-garcia/"+script
    print(f"Executing {script}...")
    result = subprocess.run(["python", sc], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"Error in {script}:\n{result.stderr}")
    
    print(f"Finished executing {script}\n")

print("All scripts executed successfully.")
