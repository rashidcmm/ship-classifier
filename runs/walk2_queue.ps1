# Lesson 27: the feature walk again with its four weak points fixed (src\feature_walk2.py). One job at a time.
# Stops nothing: waits until minimal_queue.ps1 (distance / speed runs) has finished.
# Progress: runs\master_queue.log, details in runs\walk2.log
$py = "D:\shipsound\.venv\Scripts\python.exe"
$log = "D:\shipsound\runs\master_queue.log"
Set-Location D:\shipsound
function Note($text) { "$(Get-Date -Format 'dd MMM HH:mm')  $text" | Out-File -Append -Encoding utf8 $log }
Note "walk2 queue: waiting for minimal_queue.ps1"
while (Get-Process -Id 17200 -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
Note "start  walk2 (LR judge, GB judge, tests of both final sets)"
& $py -u src\feature_walk2.py *> "D:\shipsound\runs\walk2.log"
Note "finish walk2 (exit $LASTEXITCODE)"
