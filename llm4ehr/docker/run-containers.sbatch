#!/bin/bash -l
#SBATCH --job-name=llm4ehr-dev
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=llm4ehr-%j.log
#SBATCH --error=llm4ehr-%j.err

# Load modules
module load apptainer

# Set container directory
CONTAINER_DIR="$HOME/llm4ehr-containers"
WORK_DIR="/home/hpc/iwbn/iwbn111h/llm-driven-clinical-study-analysis"

# Start services in background
echo "Starting Qdrant database..."
apptainer run "$CONTAINER_DIR/qdrant.sif" &
QDRANT_PID=$!
sleep 3

echo "Starting Backend API..."
apptainer run "$CONTAINER_DIR/backend.sif" &
BACKEND_PID=$!
sleep 2

echo "Starting Frontend..."
apptainer run "$CONTAINER_DIR/frontend.sif" &
FRONTEND_PID=$!

echo ""
echo "========================================="
echo "All services started successfully!"
echo "========================================="
echo "Qdrant PID: $QDRANT_PID"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Qdrant API: http://localhost:6333"
echo "Backend API: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="

# Wait for all processes
wait

echo "All services stopped"
