Write-Host "Starting unbreakable training marathon..." -ForegroundColor Green

while ($true) {
    python train_agent.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Training successfully completed!" -ForegroundColor Green
        break
    }

    if ($LASTEXITCODE -eq 130) {
        Write-Host "Training stopped by user. Marathon will not restart." -ForegroundColor Yellow
        break
    }

    Write-Host "Training exited with code $LASTEXITCODE. Restarting in 5 seconds..." -ForegroundColor Red
    Write-Host "Check the preceding traceback/output; this may be socket exhaustion, dependency failure, or a native crash." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
