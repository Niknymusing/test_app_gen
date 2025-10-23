import pytest
import requests
import subprocess
import time
import os
from pathlib import Path
from dotenv import load_dotenv

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
    
    # Print .env content for debugging
    env_path = PROJECT_ROOT.parent / ".env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print("ANTHROPIC_API_KEY from env:", os.getenv("ANTHROPIC_API_KEY"))
        with open(env_path, "r") as f:
            env_content = f.read()
        print(".env Content:\\n", env_content)
    else:
        print("No .env file found at", env_path)
    
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

def test_e2e_filesystem_operations(agent_service):
    """
    E2E test for the agent's filesystem operations within Docker.
    """
    # Define the task for the agent
    user_request = (
        "Write a Python script named 'fs_test.py' that creates a new text file named 'test.txt' in the workspace, "
        "writes 'Hello, MEU Agent!' to it, then reads the file and logs the content to 'read_log.txt'. "
        "Finally, execute the script."
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
    assert os.path.exists(script_path), f"Expected file {script_path} does not exist"
    with open(script_path, "r") as f:
        content = f.read()
        assert "open('test.txt', 'w')" in content or "write" in content, "Expected file creation in script"

    # 2. Check for test.txt
    test_path = PROJECT_ROOT / "test.txt"
    assert os.path.exists(test_path), f"Expected file {test_path} does not exist"
    with open(test_path, "r") as f:
        content = f.read()
        assert "Hello, MEU Agent!" in content, f"Expected content not found in {test_path}"
