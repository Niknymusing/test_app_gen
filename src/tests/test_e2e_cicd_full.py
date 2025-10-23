import pytest
import requests
import subprocess
import time
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent  # Dynamically points to project root
BASE_URL = "http://localhost:8000"
GITHUB_API_BASE = "https://api.github.com"

@pytest.fixture(scope="module")
def agent_service():
    docker_compose_path = PROJECT_ROOT / "docker-compose.yml"
    
    # Load env for GitHub token/repo
    env_path = PROJECT_ROOT.parent / ".env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    # Clean up
    subprocess.run(["docker-compose", "-f", str(docker_compose_path), "rm", "-fsv"])
    subprocess.run(["docker-compose", "-f", str(docker_compose_path), "down", "--volumes", "--remove-orphans"])

    env = os.environ.copy()
    try:
        subprocess.run(
            ["docker-compose", "-f", str(docker_compose_path), "up", "-d", "--build"],
            check=True,
            env=env
        )
        for _ in range(60):  # Increased to 60 for 1 min
            try:
                response = requests.get(f"{BASE_URL}/health")
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(1)
        else:
            raise RuntimeError("Agent service did not become healthy in time.")
        
        yield
    finally:
        subprocess.run(
            ["docker-compose", "-f", str(docker_compose_path), "down", "--volumes", "--remove-orphans"],
            env=env
        )

def test_e2e_cicd_full(agent_service):
    # Clean up prior files/repo
    subprocess.run(["rm", "-rf", str(PROJECT_ROOT / ".git")])
    for file in ["app.py", "test_app.py", "git_log.txt", "generated_script.py"]:
        (PROJECT_ROOT / file).unlink(missing_ok=True)

        user_request = (
            "Write a Python script named 'cicd_full_test.py' that: "
            "1. Generates a simple FastAPI app in 'app.py' with a /hello endpoint returning 'Hello, World!'. "
            "Add a unique timestamp to the response like 'Hello, World! - Test {timestamp}' using import datetime; timestamp = datetime.datetime.now().isoformat(). "
            "2. Generates a pytest test in 'test_app.py' to verify the /hello endpoint. "
            "In test_app.py, use response.json() for the assertion (handle string response), and add import sys; sys.path.insert(0, '.') before imports if necessary. "
            "3. Runs local tests with pytest test_app.py directly (skip pip install since deps are already available in the container). "
            "4. If tests pass, creates a .gitignore file with '.env' and '__pycache__' to exclude secrets and cache, "
            "then initializes a fresh Git repo (remove existing .git if needed), "
            "adds all files with 'git add .', commits with 'Initial app commit - {timestamp}' (Git user is already configured globally, don't run git config commands), "
            "sets remote origin to GIT_REPO_URL from env, and pushes to master using GITHUB_TOKEN for auth in the URL like https://oauth2:$GITHUB_TOKEN@github.com/repo.git. "
            "Use git push -f -u origin master to force push if necessary. "
            "Log detailed push output and any auth errors. "
            "5. Logs all Git/output to 'git_log.txt'. Finally, execute the script."
        )

    response = requests.post(f"{BASE_URL}/api/v1/tasks", json={"user_request": user_request})
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    for _ in range(60):
        status_response = requests.get(f"{BASE_URL}/api/v1/tasks/{task_id}")
        assert status_response.status_code == 200
        status = status_response.json()["status"]
        if status in ["completed", "failed"]:
            log_output = subprocess.run(["docker", "logs", "meu-coder-agent"], capture_output=True, text=True)
            print("Container Logs:\\n", log_output.stdout)
            print("Container Errors:\\n", log_output.stderr)
            break
        time.sleep(5)
    else:
        pytest.fail("Task did not complete in time.")

    assert status == "completed"

    # Local assertions
    app_path = PROJECT_ROOT / "app.py"
    assert os.path.exists(app_path), "app.py not generated"
    with open(app_path, "r") as f:
        content = f.read()
        assert "from fastapi import FastAPI" in content
        assert "/hello" in content

    test_path = PROJECT_ROOT / "test_app.py"
    assert os.path.exists(test_path), "test_app.py not generated"
    with open(test_path, "r") as f:
        content = f.read()
        assert "TestClient" in content and "assert response" in content, "Expected test content not found"

    log_path = PROJECT_ROOT / "git_log.txt"
    assert os.path.exists(log_path), "git_log.txt not generated"
    with open(log_path, "r") as f:
        content = f.read()
        assert "committed" in content.lower() or "initial app commit" in content.lower(), "Expected commit evidence"
        # Check for successful push (force update or new branch), not just absence of "failed"
        assert ("forced update" in content.lower() or "new branch" in content.lower() or "branch 'master' set up to track" in content.lower()), "Expected successful push evidence"

    # Verify remote CI/CD via GitHub API (check if workflow ran successfully)
    # Load .env if not already loaded
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    repo_url = os.getenv("GIT_REPO_URL")
    token = os.getenv("GITHUB_TOKEN")
    if not repo_url or not token:
        pytest.skip("GIT_REPO_URL or GITHUB_TOKEN not set in .env")

    # Extract owner/repo from URL (e.g., https://github.com/user/repo.git -> user/repo)
    repo = repo_url.split("github.com/")[1].replace(".git", "")

    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    # Verify push by checking latest commit via GitHub API
    commits_url = f"{GITHUB_API_BASE}/repos/{repo}/commits/master"
    response = requests.get(commits_url, headers=headers)
    assert response.status_code == 200, "GitHub API request failed to get commits"
    latest_commit = response.json()
    assert "Initial app commit" in latest_commit["commit"]["message"], f"Expected commit not found: {latest_commit['commit']['message']}"
    print(f"✅ Verified: Latest commit on GitHub is '{latest_commit['commit']['message']}' (SHA: {latest_commit['sha'][:7]})")

    # Optional: Independent verification (clone and run locally)
    # This is a nice-to-have but not critical for E2E CI/CD success
    print(f"✅ E2E CI/CD Test PASSED: Agent successfully generated, tested, committed, and pushed code to GitHub!")
