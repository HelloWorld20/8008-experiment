import subprocess
try:
    process = subprocess.run(
        ['uv', 'run', 'python', '-X', 'faulthandler', 'CNN+LSTM/learning-pytorch-lstm-deep-learning-with-m5-data.py'],
        capture_output=True,
        text=True,
        timeout=60
    )
    print("STDOUT:", process.stdout[-1000:])
    print("STDERR:", process.stderr[-1000:])
except subprocess.TimeoutExpired as e:
    print("STDOUT:", e.stdout[-1000:] if e.stdout else "None")
    print("STDERR:", e.stderr[-1000:] if e.stderr else "None")
