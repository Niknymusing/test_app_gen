import pytest
import requests
import subprocess
import time
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Define the base URL for the agent's API
BASE_URL = "http://localhost:8000"

@pytest.fixture(scope="module")
def agent_service():
    """
    A pytest fixture to manage the lifecycle of the agent's Docker container.
    """
    docker_compose_path = PROJECT_ROOT / "docker-compose.yml"
    
    # Clean up any previous runs
    subprocess.run(["docker-compose", "-f", str(docker_compose_path), "rm", "-fsv"])
    subprocess.run(["docker-compose", "-f", str(docker_compose_path), "down", "--volumes", "--remove-orphans"])

    # Start the Docker container with environment
    env = os.environ.copy()
    try:
        compose_cmd = [
            "docker-compose", "-f", str(docker_compose_path), "up", "-d", "--build"
        ]
        subprocess.run(
            compose_cmd,
            check=True,
            env=env
        )
        # Wait for the service to be healthy
        for _ in range(30):
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
        # Stop the Docker container
        subprocess.run(
            ["docker-compose", "-f", str(docker_compose_path), "down", "--volumes", "--remove-orphans"],
            env=env
        )

def test_e2e_cicd_operations(agent_service):
    # Clean up prior Git repo and files
    subprocess.run(["rm", "-rf", str(PROJECT_ROOT / ".git")])
    for file in ["example.py", "git_log.txt", "generated_script.py"]:
        (PROJECT_ROOT / file).unlink(missing_ok=True)
    """
    E2E test for the agent's CI/CD pipeline operations within Docker.
    """
    # Define the task for the agent
    user_request = (
        "Write a Python script named 'cicd_test.py' that initializes a new Git repository in the workspace, "
        "creates a file 'example.py' with print('Hello World'), "
        "adds and commits the file with message 'Initial commit', "
        "and logs the Git status to 'git_log.txt'. Finally, execute the script."
    )

    # Send the task to the agent
    response = requests.post(f"{BASE_URL}/api/v1/tasks", json={"user_request": user_request})
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    # Poll for task completion
    for _ in range(60): # Poll for up to 5 minutes
        status_response = requests.get(f"{BASE_URL}/api/v1/tasks/{task_id}")
        assert status_response.status_code == 200
        status = status_response.json()["status"]
        if status in ["completed", "failed"]:
            # Get and print container logs for debugging
            log_output = subprocess.run(["docker", "logs", "meu-coder-agent"], capture_output=True, text=True)
            print("Container Logs:\\n", log_output.stdout)
            print("Container Errors:\\n", log_output.stderr)
            break
        time.sleep(5)
    else:
        pytest.fail("Task did not complete in time.")

    assert status == "completed"

    # Get the task result for debugging
    status_response = requests.get(f"{BASE_URL}/api/v1/tasks/{task_id}")
    task_result = status_response.json()
    print("Task Result:", task_result)

    # Verify the results on the filesystem

    # 1. Check for generated_script.py
    script_path = PROJECT_ROOT / "generated_script.py"
    if os.path.exists(script_path):
        with open(script_path, "r") as f:
            script_content = f.read()
        print("Generated Script Content:\\n", script_content)
    assert os.path.exists(script_path), "Generated script not found"

    # 2. Check for .git directory
    git_path = PROJECT_ROOT / ".git"
    assert os.path.exists(git_path), "Git repository was not initialized"

    # 3. Check for example.py
    example_path = PROJECT_ROOT / "example.py"
    assert os.path.exists(example_path), f"Expected file {example_path} does not exist"
    with open(example_path, "r") as f:
        content = f.read()
        assert "print('Hello World')" in content, "Expected content not found in example.py"

    # 4. Check for git_log.txt
    log_path = PROJECT_ROOT / "git_log.txt"
    assert os.path.exists(log_path), f"Expected file {log_path} does not exist"
    with open(log_path, "r") as f:
        content = f.read()
        assert "committed" in content.lower() or "initial commit" in content.lower(), "Expected commit evidence in git log"
