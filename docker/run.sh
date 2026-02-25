#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# Build base image if not exists
if ! docker image inspect auv-base:dev &>/dev/null; then
    echo "Building base image (one-time)..."
    docker build -t auv-base:dev -f Dockerfile.dev-base ..
fi

xhost +local:docker 2>/dev/null || true

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -Eq "^auv-dev$"; then
    echo "Container exists. Starting if stopped..."
    docker start auv-dev >/dev/null 2>&1 || true
else
    echo "Creating new container..."

    # Build optional volume args
    EXTRA_VOLS=""
    [ -f "$HOME/.zsh_history" ] && EXTRA_VOLS="$EXTRA_VOLS -v $HOME/.zsh_history:/root/.zsh_history_dir/.zsh_history"
    [ -f "$HOME/.p10k.zsh" ] && EXTRA_VOLS="$EXTRA_VOLS -v $HOME/.p10k.zsh:/root/.p10k.zsh:ro"

    docker compose up -d

    # Apply optional mounts via docker cp if files exist (workaround for compose)
    if [ -f "$HOME/.p10k.zsh" ]; then
        docker cp "$HOME/.p10k.zsh" auv-dev:/root/.p10k.zsh 2>/dev/null || true
    fi
fi

echo "Connecting to container..."
docker exec -it auv-dev zsh
