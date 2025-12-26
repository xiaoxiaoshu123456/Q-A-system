param(
  [string]$Tag = "qwsystem-app:latest"
)
docker build -t $Tag -f docker/Dockerfile .
