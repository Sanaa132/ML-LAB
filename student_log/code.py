import re
from datetime import datetime

class Student:
    def __init__(self, sid, name):
        self.sid = sid
        self.name = name
        self.login_count = 0
        self.submit_count = 0
        self.is_logged_in = False
        self.abnormal = False

    def add_activity(self, activity):
        if activity == "LOGIN":
            # If already logged in, flag as abnormal behavior
            if self.is_logged_in:
                self.abnormal = True
            self.login_count += 1
            self.is_logged_in = True
        elif activity == "LOGOUT":
            self.is_logged_in = False
        elif activity == "SUBMIT_ASSIGNMENT":
            self.submit_count += 1

    def display(self):
        print(f"Student ID: {self.sid}")
        print(f"Name: {self.name}")
        print(f"Total Logins: {self.login_count}")
        print(f"Total Submissions: {self.submit_count}")
        if self.abnormal:
            print("⚠ Abnormal Behavior: Multiple logins without logout detected")
        print("-" * 30)

# --- Configuration ---
LOG_FILE = "student_log.txt"
REPORT_FILE = "activity_report.txt"

# --- Phase 1: Logging Activities ---
print("\n=== REAL-TIME STUDENT ACTIVITY LOGGER ===")
print("Allowed activities: LOGIN, LOGOUT, SUBMIT_ASSIGNMENT")
print("Type 'stop' as Student ID to finish\n")

with open(LOG_FILE, "a") as file:
    while True:
        sid = input("Student ID (e.g., S101): ").strip()
        if sid.lower() == "stop":
            break
        
        name = input("Student Name: ").strip()
        activity = input("Activity: ").upper().strip()

        try:
            # Validate Student ID format (S followed by digits)
            if not re.fullmatch(r"S\d+", sid):
                raise ValueError("Invalid Student ID format. Use 'S' followed by numbers.")
            
            # Validate Activity type
            if activity not in ["LOGIN", "LOGOUT", "SUBMIT_ASSIGNMENT"]:
                raise ValueError("Invalid Activity. Use LOGIN, LOGOUT, or SUBMIT_ASSIGNMENT.")

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            # Store in log file
            file.write(f"{sid} | {name} | {activity} | {date_str} | {time_str}\n")
            print("✔ Activity logged successfully!\n")
            
        except ValueError as e:
            print(f"❌ Error: {e}\n")

# --- Phase 2: Analysis (Generator Function) ---
def read_logs(filename):
    """Generator to read and yield valid log entries."""
    try:
        with open(filename, "r") as file:
            for line in file:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 5:
                    sid, name, activity, date, time = parts
                    yield sid, name, activity, date
    except FileNotFoundError:
