#!/bin/bash
set -e

# Configure Git from environment variables
if [ -n "$GIT_USER_NAME" ]; then
    git config --global user.name "$GIT_USER_NAME"
fi

if [ -n "$GIT_USER_EMAIL" ]; then
    git config --global user.email "$GIT_USER_EMAIL"
fi

# Configure Git credential helper to use GITHUB_TOKEN from environment
# This allows Git to automatically authenticate for clone/push operations
if [ -n "$GITHUB_TOKEN" ]; then
    # Set up credential helper that returns the token
    git config --global credential.helper 'cache --timeout=3600'
    
    # For HTTPS URLs, configure Git to use the token
    git config --global url."https://oauth2:${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
    
    # Set credentials for GitHub
    echo "https://oauth2:${GITHUB_TOKEN}@github.com" > /tmp/.git-credentials
    git config --global credential.helper "store --file=/tmp/.git-credentials"
fi

# Configure safe directory for Git operations in /workspace
git config --global --add safe.directory /workspace
git config --global --add safe.directory '*'

# Set default branch to master (suppress hints)
git config --global init.defaultBranch master

# Execute the main command
exec "$@"

