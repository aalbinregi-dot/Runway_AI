@echo off
setlocal EnableDelayedExpansion

REM ===========================================================================
REM  Runway Performance AI - launcher
REM
REM  Double-click for a menu, or call a command directly:
REM      run.bat info | test | analyze <file> | train | rebuild | verify
REM      run.bat dashboard | api | sensitivity | setup | clean
REM
REM  Handles two things that are easy to get wrong by hand:
REM    * PYTHONPATH must point at src\ or "python -m runway_ai" will not resolve
REM    * video input needs the 3.12 environment, everything else does not
REM ===========================================================================

REM Work from this file's directory, whatever it was launched from.
cd /d "%~dp0"

set "PYTHONPATH=%~dp0src"
set "PY=python"
set "PYVID="

REM Prefer a local analysis venv if one exists.
if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"

REM A separate environment is only needed for reading video files, because
REM MediaPipe pins Python to 3.9-3.12. Either name is accepted.
if exist ".venv311\Scripts\python.exe" set "PYVID=%~dp0.venv311\Scripts\python.exe"
if exist ".venv312\Scripts\python.exe" set "PYVID=%~dp0.venv312\Scripts\python.exe"

set "INTERACTIVE="
if "%~1"=="" set "INTERACTIVE=1"
if defined INTERACTIVE goto MENU
goto DISPATCH


REM ===========================================================================
:MENU
cls
echo.
echo   ===========================================================
echo    RUNWAY PERFORMANCE AI
echo   ===========================================================
echo.
echo    CHECK
echo      1.  System status          (models loaded, backends)
echo      2.  Run test suite         (Python + browser, ~40s)
echo      3.  Score the sample file  (the supplied workbook)
echo      4.  Show model results     (per-criterion accuracy)
echo      5.  Sanity check           (expert vs novice ranking)
echo.
echo    RUN
echo      6.  Analyse a file...      (video, csv or xlsx)
echo      7.  Analyse a group...     (everyone in one clip)
echo      8.  Annotated video        (skeleton + live metrics)
echo      9.  Web app                (React + API, both servers)
echo     10.  REST API               (FastAPI + docs)
echo.
echo    BUILD
echo     11.  Full rebuild           (data + rubric + train, ~5 min)
echo     12.  Feature sensitivity    (validation sweep, ~10 min)
echo     13.  Install dependencies
echo     14.  Clean generated files
echo.
echo      0.  Exit
echo.
set "CHOICE="
set /p "CHOICE=   Select: "

if "%CHOICE%"=="1"  goto INFO
if "%CHOICE%"=="2"  goto TEST
if "%CHOICE%"=="3"  goto SAMPLE
if "%CHOICE%"=="4"  goto RESULTS
if "%CHOICE%"=="5"  goto SANITY
if "%CHOICE%"=="6"  goto ASKFILE
if "%CHOICE%"=="7"  goto ASKGROUP
if "%CHOICE%"=="8"  goto ASKRENDER
if "%CHOICE%"=="9"  goto WEB
if "%CHOICE%"=="10" goto API
if "%CHOICE%"=="11" goto REBUILD
if "%CHOICE%"=="12" goto SENSITIVITY
if "%CHOICE%"=="13" goto SETUP
if "%CHOICE%"=="14" goto CLEAN
if "%CHOICE%"=="0"  goto END
echo.
echo   Not an option.
timeout /t 2 >nul
goto MENU


REM ===========================================================================
:DISPATCH
set "CMD=%~1"
if /i "%CMD%"=="info"        goto INFO
if /i "%CMD%"=="diagnose"    goto DIAGNOSE
if /i "%CMD%"=="diagnostics" goto DIAGNOSE
if /i "%CMD%"=="test"        goto TEST
if /i "%CMD%"=="tests"       goto TEST
if /i "%CMD%"=="sample"      goto SAMPLE
if /i "%CMD%"=="results"     goto RESULTS
if /i "%CMD%"=="verify"      goto VERIFY
if /i "%CMD%"=="sanity"      goto SANITY
if /i "%CMD%"=="analyze"     goto ANALYZE_ARG
if /i "%CMD%"=="analyse"     goto ANALYZE_ARG
if /i "%CMD%"=="group"       goto GROUP_ARG
if /i "%CMD%"=="render"      goto RENDER_ARG
if /i "%CMD%"=="web"         goto WEB
if /i "%CMD%"=="api"         goto API
if /i "%CMD%"=="train"       goto TRAIN
if /i "%CMD%"=="rebuild"     goto REBUILD
if /i "%CMD%"=="sensitivity" goto SENSITIVITY
if /i "%CMD%"=="setup"       goto SETUP
if /i "%CMD%"=="clean"       goto CLEAN
if /i "%CMD%"=="help"        goto HELP
if /i "%CMD%"=="-h"          goto HELP
if /i "%CMD%"=="--help"      goto HELP

echo Unknown command: %CMD%
echo.
goto HELP


REM ===========================================================================
:HELP
echo.
echo   Usage: run.bat [command]
echo.
echo     info           system status: models, rubric, video backend
echo     diagnose       full capability report, with a fix for each gap
echo     test           run the Python suite and the browser tests
echo     sample         score the supplied workbook
echo     results        print per-criterion accuracy
echo     sanity         check expert/intermediate/novice ranking
echo     verify         info + test + sample + sanity, in one pass
echo     analyze FILE   score a video, csv or xlsx  (extra flags pass through)
echo     group FILE     find and measure everyone in one clip
echo     render FILE    annotated video: skeleton, live metrics, scores
echo     web            start the React app and the API together
echo     api            launch the FastAPI service
echo     rebuild        regenerate data, recalibrate rubric, retrain
echo     sensitivity    feature validation sweep
echo     setup          pip install -r requirements.txt
echo     clean          delete generated data, models and reports
echo.
echo   Run with no arguments for an interactive menu.
echo.
goto END


REM ===========================================================================
:INFO
echo.
echo   --- System status ---------------------------------------
echo.
"%PY%" -m runway_ai info
if defined PYVID (
    echo.
    echo   Video pipeline ready: !PYVID!
) else (
    echo.
    echo   No video environment. MediaPipe needs Python 3.9-3.12:
    echo       py -3.11 -m venv .venv311
    echo       .venv311\Scripts\pip install -r requirements-video.txt
)
echo.
echo   For a full capability report with a fix for each gap:
echo       run.bat diagnose
goto DONE


REM ===========================================================================
:DIAGNOSE
echo.
echo   --- Diagnostics -----------------------------------------
echo.
REM Runs the same report the Guide page shows, so a user without the web app
REM running can still produce something worth pasting into a bug report.
"%PY%" -c "import sys; sys.path.insert(0,'app'); import diagnostics; print(diagnostics.as_text())"
goto DONE


REM ===========================================================================
:TEST
echo.
echo   --- Test suite ------------------------------------------
echo.
"%PY%" -m pytest tests -q
set "PYFAIL=!errorlevel!"

REM The live scene classifier is browser-side JavaScript, so pytest cannot reach
REM it. It runs on node's own test runner with no dependencies - but only if the
REM web app's node_modules exist, since a fresh checkout has not run npm install
REM and a missing toolchain must not read as a test failure.
REM
REM Written with labels rather than parenthesised if/else blocks on purpose: a
REM closing bracket inside an echo ends the block early, which is how this
REM printed "--------------- was unexpected at this time."
if not exist "web\node_modules" goto TEST_NOJS
echo.
echo   --- Live scene classifier - node --test -----------------
echo.
pushd web
call npm test --silent
set "JSFAIL=!errorlevel!"
popd
goto TEST_VERDICT

:TEST_NOJS
set "JSFAIL=0"
echo.
echo   Skipping the browser tests - run "run.bat web" once to install them.

:TEST_VERDICT
if not "!PYFAIL!!JSFAIL!"=="00" goto TEST_FAILED
echo.
echo   All tests passed.
goto DONE

:TEST_FAILED
echo.
echo   TESTS FAILED.
goto DONE


REM ===========================================================================
:SAMPLE
echo.
echo   --- Scoring the supplied workbook -----------------------
echo.
if not exist "Synthetic_Runway_Keypoints_Sample.xlsx" (
    echo   Sample workbook not found in %CD%
    goto DONE
)
"%PY%" -m runway_ai analyze "Synthetic_Runway_Keypoints_Sample.xlsx"
goto DONE


REM ===========================================================================
:RESULTS
echo.
echo   --- Model accuracy (all out-of-fold) --------------------
echo.
if not exist "reports\model_comparison.csv" (
    echo   No results yet. Run option 11 ^(full rebuild^) first.
    goto DONE
)
"%PY%" scripts\show_results.py
goto DONE


REM ===========================================================================
:SANITY
echo.
echo   --- Does it rank skill correctly? -----------------------
echo.
"%PY%" scripts\sanity_check.py
goto DONE


REM ===========================================================================
:VERIFY
call :RUNSTEP "System status"  "%PY%" -m runway_ai info
call :RUNSTEP "Test suite"     "%PY%" -m pytest tests -q
call :RUNSTEP "Sanity check"   "%PY%" scripts\sanity_check.py
echo.
echo   --- Verification complete -------------------------------
goto DONE


REM ===========================================================================
:ASKFILE
echo.
set "TARGET="
set /p "TARGET=   Path to video/csv/xlsx (or drag the file here): "
REM Dragging a file into the console wraps it in quotes; %~ strips them.
set "TARGET=%TARGET:"=%"
if "%TARGET%"=="" goto MENU
if not exist "%TARGET%" (
    echo.
    echo   File not found: %TARGET%
    goto DONE
)
set "EXTRA=--save-report"
goto RUN_ANALYZE


REM ===========================================================================
:GROUP_ARG
if "%~2"=="" (
    echo.
    echo   Usage: run.bat group ^<video^>
    goto END
)
set "TARGET=%~2"
if not exist "%TARGET%" (
    echo   File not found: %TARGET%
    goto END
)
goto RUN_GROUP


REM ===========================================================================
:ASKGROUP
echo.
set "TARGET="
set /p "TARGET=   Path to a group video (or drag the file here): "
set "TARGET=%TARGET:"=%"
if "%TARGET%"=="" goto MENU
if not exist "%TARGET%" (
    echo.
    echo   File not found: %TARGET%
    goto DONE
)

:RUN_GROUP
if not defined PYVID (
    echo.
    echo   Group analysis reads video, which needs MediaPipe.
    echo   MediaPipe supports Python 3.9-3.12 only. Create that environment:
    echo.
    echo       py -3.11 -m venv .venv311
    echo       .venv311\Scripts\pip install -r requirements-video.txt
    echo.
    goto DONE
)
echo.
echo   Finding everyone with a visible face, then measuring each separately.
echo   Pose estimation runs once per person, so this takes longer than a solo
echo   clip. The web app (option 9) also scores and ranks them.
echo.
REM Runs in the video interpreter: MediaPipe is not installed in the analysis
REM environment, which is where the trained models live.
"%PYVID%" -m runway_ai extract-multi "%TARGET%"
goto DONE


REM ===========================================================================
:ANALYZE_ARG
if "%~2"=="" (
    echo.
    echo   Usage: run.bat analyze ^<file^> [--user NAME] [--save-report] [--json]
    goto END
)
set "TARGET=%~2"
REM A share link is a documented input - the README advertises
REM `run.bat analyze "https://youtu.be/..."` - and the pipeline fetches it
REM server-side. The existence test below would reject it as a missing file, so
REM links skip straight to the CLI in the main environment, which is the one with
REM yt-dlp.
set "ISLINK="
echo %TARGET% | findstr /i /b /c:"http://" /c:"https://" >nul && set "ISLINK=1"
if defined ISLINK goto LINK_ANALYZE
if not exist "%TARGET%" (
    echo   File not found: %TARGET%
    goto END
)
REM Collect any remaining flags and pass them straight through to the CLI.
set "EXTRA="
shift
shift
:COLLECT
if "%~1"=="" goto RUN_ANALYZE
set "EXTRA=%EXTRA% %1"
shift
goto COLLECT


REM ===========================================================================
REM  A link: fetched and scored by the CLI in the main environment.
:LINK_ANALYZE
set "EXTRA="
shift
shift
:COLLECT_LINK
if "%~1"=="" goto DO_LINK
set "EXTRA=%EXTRA% %1"
shift
goto COLLECT_LINK

:DO_LINK
echo.
echo   --- Analysing link: %TARGET% ---
echo.
"%PY%" -m runway_ai analyze "%TARGET%"%EXTRA%
goto DONE


REM ===========================================================================
:RUN_ANALYZE
set "RUNNER=%PY%"
set "ISVIDEO="
for %%F in ("%TARGET%") do set "EXT=%%~xF"
for %%V in (.mp4 .mov .avi .mkv .webm .m4v) do if /i "!EXT!"=="%%V" set "ISVIDEO=1"

if not defined ISVIDEO goto SCORE_DIRECT

if not defined PYVID (
    echo.
    echo   "%TARGET%" is a video, which needs MediaPipe.
    echo   MediaPipe supports Python 3.9-3.12 only. Create that environment:
    echo.
    echo       py -3.11 -m venv .venv311
    echo       .venv311\Scripts\pip install -r requirements-video.txt
    echo.
    echo   run.bat picks it up automatically once it exists.
    goto DONE
)

REM Step 1: pose estimation in the video environment.
REM Step 2: scoring in the main one. Splitting them means the trained models
REM are never unpickled against a different scikit-learn build.
REM Keypoints land in data\interim so they can be inspected, kept, or
REM re-scored without repeating the slow extraction step.
echo.
echo   --- Step 1/2: extracting pose from %TARGET% ---
echo.
if not exist "data\interim" mkdir "data\interim"
set "KPFILE=%~dp0data\interim\_last_video.csv"

"!PYVID!" -m runway_ai extract "%TARGET%" -o "!KPFILE!"
if errorlevel 1 (
    echo.
    echo   Pose extraction failed. Common causes: the whole body is not in
    echo   frame, the clip is too short, or the subject is too dark.
    goto DONE
)

REM The frame rate travels in a .meta.json sidecar written by `extract`,
REM so there is nothing to parse out of stdout here.
echo.
echo   --- Step 2/2: scoring ---
"%PY%" -m runway_ai analyze "!KPFILE!"%EXTRA%
goto DONE

:SCORE_DIRECT
echo.
echo   --- Analysing: %TARGET% ---
echo.
"%RUNNER%" -m runway_ai analyze "%TARGET%"%EXTRA%
goto DONE


REM ===========================================================================
:ASKRENDER
echo.
set "RTARGET="
set /p "RTARGET=   Path to the video (or drag it here): "
set "RTARGET=%RTARGET:"=%"
if "%RTARGET%"=="" goto MENU
goto DO_RENDER


REM ===========================================================================
:RENDER_ARG
if "%~2"=="" (
    echo.
    echo   Usage: run.bat render ^<video^>
    goto END
)
set "RTARGET=%~2"

:DO_RENDER
if not exist "%RTARGET%" (
    echo.
    echo   File not found: %RTARGET%
    goto DONE
)
if not defined PYVID (
    echo.
    echo   Rendering needs pose data, which needs MediaPipe. Create the
    echo   video environment first:
    echo       py -3.11 -m venv .venv311
    echo       .venv311\Scripts\pip install -r requirements-video.txt
    goto DONE
)
if not exist "data\interim" mkdir "data\interim"
set "KPFILE=%~dp0data\interim\_last_video.csv"
echo.
echo   --- Step 1/2: extracting pose ---
echo.
"!PYVID!" -m runway_ai extract "%RTARGET%" -o "!KPFILE!"
if errorlevel 1 goto DONE
echo.
echo   --- Step 2/2: rendering annotated video ---
echo.
"%PY%" -m runway_ai render "%RTARGET%" -k "!KPFILE!"
goto DONE


REM ===========================================================================
:WEB
"%PY%" scripts\check_env.py api
if errorlevel 1 goto DONE
where node >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Node.js is required for the web app. Install it from nodejs.org,
    echo   then run this option again.
    goto DONE
)
REM Whether web\node_modules can run *here*, not merely whether it exists.
REM Rolldown and oxlint ship their native code as per-platform optional packages,
REM so a tree copied from a Mac contains only the darwin bindings and `npm run
REM dev` fails with "Cannot find module @rolldown/binding-win32-x64-msvc" - while
REM `npm test` still passes, because node --test needs no binding.
REM Matched with a trailing wildcard: the binding packages carry an ABI suffix
REM that platform+arch alone does not predict (win32-x64-msvc, linux-x64-gnu,
REM but plain darwin-arm64), so an exact-name test would fail even when the
REM correct binding is installed.
set "WEBDEPS=0"
set "NODEPA="
if exist "web\node_modules" (
    for /f "delims=" %%P in ('node -p "process.platform+'-'+process.arch" 2^>nul') do set "NODEPA=%%P"
)
if defined NODEPA (
    if exist "web\node_modules\@rolldown\binding-!NODEPA!*" set "WEBDEPS=1"
)
if "%WEBDEPS%"=="0" (
    echo.
    echo   Installing web dependencies for this platform...
    pushd web
    if exist "package-lock.json" (call npm ci) else (call npm install)
    popd
)
echo.
echo   Starting both servers.
echo     API  http://127.0.0.1:8000
echo     App  http://localhost:5180
echo.
echo   The API opens in a second window. Close it when you are done.
echo   Press Ctrl+C here to stop the web app.
echo.
start "Runway AI - API" /min "%PY%" -m uvicorn app.api:app --host 127.0.0.1 --port 8000
pushd web
call npm run dev
popd
goto DONE


REM ===========================================================================
:API
"%PY%" scripts\check_env.py api
if errorlevel 1 goto DONE
echo.
echo   Starting FastAPI. Interactive docs:  http://127.0.0.1:8000/docs
echo   Press Ctrl+C in this window to stop.
echo.
"%PY%" -m uvicorn app.api:app --reload
goto DONE


REM ===========================================================================
:TRAIN
call :RUNSTEP "Calibrating rubric" "%PY%" scripts\calibrate_rubric.py
call :RUNSTEP "Training models"    "%PY%" scripts\train_model.py
goto DONE


REM ===========================================================================
:REBUILD
echo.
echo   Regenerates the corpus, refits the rubric and retrains.
echo   Takes about 5 minutes. Existing models will be overwritten.
echo.
set "OK="
set /p "OK=   Continue? [y/N]: "
if /i not "%OK%"=="y" goto DONE

call :RUNSTEP "Generating dataset"  "%PY%" scripts\build_dataset.py --n 1500 --seed 42 --no-keypoints
call :RUNSTEP "Calibrating rubric"  "%PY%" scripts\calibrate_rubric.py
call :RUNSTEP "Training models"     "%PY%" scripts\train_model.py
call :RUNSTEP "Test suite"          "%PY%" -m pytest tests -q
echo.
echo   --- Rebuild complete ------------------------------------
echo   Seeds are fixed, so this should reproduce mean MAE 0.902 / R2 0.735.
goto DONE


REM ===========================================================================
:SENSITIVITY
echo.
echo   Sweeps each latent quality and reports which features respond.
echo   This is the check that a feature measures what it claims to.
echo   Takes about 10 minutes; output also written to reports\sensitivity.txt
echo.
"%PY%" scripts\sensitivity.py
goto DONE


REM ===========================================================================
:SETUP
echo.
echo   --- Installing dependencies -----------------------------
echo.
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo   Install failed. Check the messages above.
) else (
    echo.
    echo   Done. Video support is separate and optional:
    echo       py -3.11 -m venv .venv311
    echo       .venv311\Scripts\pip install -r requirements-video.txt
)
goto DONE


REM ===========================================================================
:CLEAN
echo.
echo   Deletes generated data, trained models and reports.
echo   Source code, docs and the supplied workbook are NOT touched.
echo.
set "OK="
set /p "OK=   Delete generated files? [y/N]: "
if /i not "%OK%"=="y" goto DONE

if exist "data\interim"   rd /s /q "data\interim"
if exist "data\processed" rd /s /q "data\processed"
if exist "data\runway.db" del /q "data\runway.db"
if exist "models"         rd /s /q "models"
if exist "reports"        rd /s /q "reports"
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
if exist ".pytest_cache"  rd /s /q ".pytest_cache"

echo.
echo   Cleaned. Run option 11 to rebuild.
goto DONE


REM ===========================================================================
REM  Helper: run one labelled step and stop the batch if it fails.
REM ===========================================================================
:RUNSTEP
echo.
echo   --- %~1 ---
echo.
shift
%1 %2 %3 %4 %5 %6 %7 %8 %9
if errorlevel 1 (
    echo.
    echo   Step failed: %~1
    exit /b 1
)
exit /b 0


REM ===========================================================================
:DONE
echo.
if defined INTERACTIVE (
    pause
    goto MENU
)

:END
endlocal
