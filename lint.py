import subprocess
import sys
import os


def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def main():
    # Determine the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))

    print("Running Black...")
    black_command = f"black {project_root}"
    black_code, black_output, black_error = run_command(black_command)
    print(black_output.decode())
    print(black_error.decode())

    print("\nRunning Flake8...")
    # Exclude the virtual environment and any other problematic directories
    flake8_command = f"flake8 {project_root} --exclude=.venv,venv,env,build,dist"
    flake8_code, flake8_output, flake8_error = run_command(flake8_command)
    print(flake8_output.decode())
    print(flake8_error.decode())

    if black_code != 0 or flake8_code != 0:
        print("\nLinting failed. Please fix the issues above.")
        sys.exit(1)
    else:
        print("\nLinting passed successfully!")


if __name__ == "__main__":
    main()
