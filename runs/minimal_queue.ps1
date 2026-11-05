# Distance and speed: is the low accuracy the model, the features, or missing information? One job at a time.
# Stops nothing: waits until overnight_queue.ps1 (with_other.py, then the ship-type CNN runs) has finished.
# Progress: runs\master_queue.log
$py = "D:\shipsound\.venv\Scripts\python.exe"
$log = "D:\shipsound\runs\master_queue.log"
Set-Location D:\shipsound
function Note($text) { "$(Get-Date -Format 'dd MMM HH:mm')  $text" | Out-File -Append -Encoding utf8 $log }
Note "minimal queue: waiting for overnight_queue.ps1 (with_other.py, then the ship-type CNN runs)"
while (Get-Process -Id 27572 -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
Note "overnight_queue.ps1 finished"

function Step($name, $logfile, [string[]]$arguments) {
    Note "start  $name"
    & $py -u @arguments *> "D:\shipsound\runs\$logfile"
    Note "finish $name (exit $LASTEXITCODE)"
}

# model: same features as gradient boosting, other models
Step "1 distance, linear"                  "linear_distance.log" @("src\distance_classical.py", "--model", "linear")
Step "2 speed, linear"                     "linear_speed.log"    @("src\distance_classical.py", "--model", "linear", "--target", "speed")
# missing information: the ship's real size (how loud it is) and the hydrophone
Step "3 distance + ship size + hydrophone" "size_distance.log"   @("src\distance_classical.py", "--size")
Step "4 speed + ship size + hydrophone"    "size_speed.log"      @("src\distance_classical.py", "--size", "--target", "speed")
# features: new features from the same passage's other clips
Step "5 distance + neighbour clips"        "nb_distance.log"     @("src\distance_classical.py", "--neighbours")
Step "6 speed + neighbour clips"           "nb_speed.log"        @("src\distance_classical.py", "--neighbours", "--target", "speed")
# model: SVM (slowest, last)
Step "7 distance, SVM (RBF)"               "svm_distance.log"    @("src\distance_classical.py", "--model", "svm")
Step "8 speed, SVM (RBF)"                  "svm_speed.log"       @("src\distance_classical.py", "--model", "svm", "--target", "speed")
Note "all done"
