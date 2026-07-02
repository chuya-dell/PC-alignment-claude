@echo off
robocopy "I:\マイドライブ\データ移動" "G:\マイドライブ\1.実験データ_gdrive\5.生データ" /E /Z /MON:1 /MOT:5 /LOG+:C:\Users\%USERNAME%\sync_log.txt /TEE
