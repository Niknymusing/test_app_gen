import os
import sys
import subprocess
import datetime
import shutil
from pathlib import Path

def write_app_file():
    app_content = """from fastapi import FastAPI
import datetime

app = FastAPI()

@app.get("/hello")
async def hello():
    timestamp = datetime.datetime.now().isoformat()
    return f"Hello, World! - Test {timestamp}"
"""
    with open('app.py', 'w') as f:
        f.write(app_content)

def write_test_file():
    test_content = """import sys
sys.path.insert(0, '.')

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_hello_endpoint():
    response = client.get("/hello")
    assert response.status_code == 200
    response_text = response.text.strip('"')  # Remove quotes from JSON string
    assert response_text.startswith("Hello, World! - Test ")
"""
    with open('test_app.py', 'w') as f:
        f.write(test_content)

def write_gitignore():
    gitignore_content = """.env
__pycache__/
*.pyc
"""
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)

def run_tests():
    try:
        result = subprocess.run(['pytest', 'test_app.py', '-v'], 
                              capture_output=True, 
                              text=True)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def setup_and_push_git():
    timestamp = datetime.datetime.now().isoformat()
    log_output = []
    
    try:
        # Remove existing .git if it exists
        if os.path.exists('.git'):
            shutil.rmtree('.git')
        
        # Initialize new git repo
        subprocess.run(['git', 'init'], check=True, capture_output=True, text=True)
        log_output.append("Git repository initialized")

        # Add all files
        subprocess.run(['git', 'add', '.'], check=True, capture_output=True, text=True)
        log_output.append("Files added to git")

        # Commit changes
        commit_msg = f"Initial app commit - {timestamp}"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True, capture_output=True, text=True)
        log_output.append(f"Changes committed: {commit_msg}")

        # Set remote origin using environment variables
        git_repo_url = os.getenv('GIT_REPO_URL')
        github_token = os.getenv('GITHUB_TOKEN')
        
        if not git_repo_url or not github_token:
            raise Exception("GIT_REPO_URL or GITHUB_TOKEN environment variables not set")

        # Construct authenticated URL
        auth_url = git_repo_url.replace('https://', f'https://oauth2:{github_token}@')
        
        # Set remote origin
        subprocess.run(['git', 'remote', 'add', 'origin', auth_url], check=True, capture_output=True, text=True)
        log_output.append("Remote origin set")

        # Force push to master
        push_result = subprocess.run(['git', 'push', '-f', '-u', 'origin', 'master'], 
                                   capture_output=True, 
                                   text=True)
        log_output.append("Push completed with output:")
        log_output.append(push_result.stdout)
        if push_result.stderr:
            log_output.append(f"Push stderr: {push_result.stderr}")

        return True, '\n'.join(log_output)
    
    except Exception as e:
        return False, f"Git operations failed: {str(e)}\n{''.join(log_output)}"

def main():
    # Create log file
    log_file = Path('git_log.txt')
    
    try:
        # Step 1: Generate FastAPI app
        write_app_file()
        print("Created app.py")
        
        # Step 2: Generate test file
        write_test_file()
        print("Created test_app.py")
        
        # Step 3: Run tests
        tests_passed, test_output = run_tests()
        print("Test output:")
        print(test_output)
        
        if not tests_passed:
            raise Exception("Tests failed, aborting git push")
        
        # Step 4: Create .gitignore and push to git
        write_gitignore()
        print("Created .gitignore")
        
        # Step 5: Setup git and push
        git_success, git_output = setup_and_push_git()
        
        # Log all output
        with open(log_file, 'w') as f:
            f.write("=== Test Output ===\n")
            f.write(test_output)
            f.write("\n=== Git Operations ===\n")
            f.write(git_output)
        
        if not git_success:
            raise Exception("Git operations failed, check git_log.txt for details")
        
        print("All operations completed successfully!")
        print(f"Check {log_file} for detailed logs")
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write("\n=== Error ===\n")
            f.write(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()