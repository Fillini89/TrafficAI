Write-Host "Starting unbreakable training marathon..." -ForegroundColor Green

while ($true) {
    python train_agent.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Training successfully completed!" -ForegroundColor Green
        break
    }
    
    Write-Host "System crash detected (Socket exhaustion). Restarting in 5 seconds..." -ForegroundColor Red
    Start-Sleep -Seconds 5
}