# VectorLab RunPod Serverless worker

This RunPod queue endpoint scales from zero to one GPU worker. Each invocation
claims at most one queued Supabase conversion job, writes the SVG result, and
then exits. It never polls while idle.

## Runtime variables

- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY` (RunPod Secret)
- `HF_TOKEN` (RunPod Secret; required by `bigcode/starcoderbase-1b`)
- Optional values documented in `.env.example`

Do not place real secret values in this directory. Map RunPod Secrets to the
runtime variables in the deployed endpoint settings.

## Build, deploy, and remove

```bash
docker build -t ghcr.io/<owner>/vectorlab-runpod-worker:<tag> .
# Create the RunPod endpoint with workersMin=0 and workersMax=1.
# Delete the endpoint from RunPod when the service is retired.
```

The endpoint must remain `workers=(0, 1)`. A minimum worker count of one would
turn this into continuously billed GPU hosting.
