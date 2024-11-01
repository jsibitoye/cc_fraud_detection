import os
import subprocess
from datetime import datetime, timedelta

# Path to your repo
repo_path = r"C:\Dev\cc_fraud_detection"
os.chdir(repo_path)

# Start and end dates
start_date = datetime(2024, 11, 1)
end_date = datetime(2024, 12, 31)
delta = timedelta(days=1)

current = start_date
while current <= end_date:
    for hour in [10, 18]:  # 10 AM and 6 PM commits
        # Write/update a dev log file
        with open("devlog.txt", "a") as f:
            f.write(f"Work update on {current.strftime('%Y-%m-%d %H:%M')}\n")

        # Stage changes
        subprocess.run(["git", "add", "."], check=True)

        # Commit with backdated timestamp
        commit_msg = f"Update on {current.strftime('%Y-%m-%d %H:%M')}"
        date_str = current.replace(hour=hour, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S")

        subprocess.run([
            "git", "commit", "-m", commit_msg,
            "--date", date_str
        ], check=True)

    current += delta

print("✅ Backdated commits created for Nov–Dec 2024 (2 per day).")
