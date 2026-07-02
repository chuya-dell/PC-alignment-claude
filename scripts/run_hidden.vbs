Set objShell = CreateObject("WScript.Shell")
batPath = "C:\Users\" & objShell.ExpandEnvironmentStrings("%USERNAME%") & "\Scripts\sync_digitalcount.bat"
objShell.Run """" & batPath & """", 0, False
