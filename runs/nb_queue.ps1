# Steps 5-6 of minimal_queue.ps1 only (neighbour-clip features), one after the other.
$py = "D:\shipsound\.venv\Scripts\python.exe"
$log = "D:\shipsound\runs\master_queue.log"
Set-Location D:\shipsound
function Note($text) { "$(Get-Date -Format 'dd MMM HH:mm')  $text" | Out-File -Append -Encoding utf8 $log }
Note "start  5 distance + neighbour clips (restart)"
& $py -u src\distance_classical.py --neighbours *> D:\shipsound\runs\nb_distance.log
Note "finish 5 distance + neighbour clips (exit $LASTEXITCODE)"
Note "start  6 speed + neighbour clips"
& $py -u src\distance_classical.py --neighbours --target speed *> D:\shipsound\runs\nb_speed.log
Note "finish 6 speed + neighbour clips (exit $LASTEXITCODE)"
