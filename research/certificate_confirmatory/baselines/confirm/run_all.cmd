@echo off
setlocal enabledelayedexpansion
rem Confirmatory run: certificate, then BASE, ROME, MEMIT, FT on the window, then the fixed analysis.
rem Resumable: finished (editor, case) pairs are skipped. See DEVIATIONS.md for the allocator setting.
set PY=C:\Users\md\AppData\Local\Programs\Python\Python311\python.exe
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /d D:\HLM5-cert-confirm\baselines\confirm
if not exist results mkdir results
if not exist logs mkdir logs
echo !date! !time! resume >> logs\progress.txt
%PY% -u cert_window.py >> logs\cert.log 2>&1
echo !date! !time! cert done >> logs\progress.txt
for %%E in (BASE ROME MEMIT FT) do (
  %PY% -u editors_window.py --editor %%E >> logs\%%E.log 2>&1
  echo !date! !time! %%E done >> logs\progress.txt
)
%PY% -u analyze_confirm.py > logs\analysis.log 2>&1
echo !date! !time! analysis done >> logs\progress.txt
