@echo off
setlocal enabledelayedexpansion

REM =====================================================
REM Battery SOH Training Script with Time Estimation
REM =====================================================

echo Starting Battery SOH Training with Multiple Configurations...
echo.

REM Base parameters (modify these if needed)
set BASE_EPOCHS=100
set BASE_BATCH_SIZE=32
set BASE_LR=1e-3
set BASE_INPUT_WINDOW=100
set BASE_NUM_CHANNELS=32 64 64
set BASE_KERNEL_SIZE=3
set BASE_NUM_RUNS=2
set BASE_MODEL_DIR=models
set BASE_WANDB_PROJECT=battery_soh
set BASE_CHUNK_SIZE=16

REM Additional flags
set ADDITIONAL_FLAGS=--use_wandb --scale_data

REM Arrays for configurations
set attention_types=chunk
set output_windows=1  

REM Time tracking variables
set total_configs=9
set run_counter=0
set /a estimated_time_per_run=0
set total_start_time=%time%

REM Convert start time to seconds for calculations
call :time_to_seconds %total_start_time% total_start_seconds

echo =====================================================
echo Configuration Summary:
echo Epochs: %BASE_EPOCHS%
echo Batch Size: %BASE_BATCH_SIZE%
echo Learning Rate: %BASE_LR%
echo Input Window: %BASE_INPUT_WINDOW%
echo Attention Types: %attention_types%
echo Output Windows: %output_windows%
echo Total Configurations: %total_configs%
echo Start Time: %total_start_time%
echo =====================================================
echo.

REM Main execution loop
for %%a in (%attention_types%) do (
    for %%w in (%output_windows%) do (
        set /a run_counter+=1
        echo.
        echo =====================================================
        echo Run !run_counter! of %total_configs%: attention_type=%%a, output_window=%%w
        
        REM Calculate and display progress
        set /a progress_percent=!run_counter!*100
        set /a progress_percent=!progress_percent!/!total_configs!
        echo Progress: !progress_percent!%% complete
        
        REM Estimate completion time based on previous runs
        if !run_counter! GTR 1 (
            call :calculate_eta !run_counter! !total_configs! !estimated_time_per_run!
        ) else (
            echo ETA: Calculating after first run...
        )
        echo =====================================================
        
        REM Record start time for this run
        set run_start_time=!time!
        call :time_to_seconds !run_start_time! run_start_seconds
        
        REM Create unique model directory for this configuration
        set current_model_dir=%BASE_MODEL_DIR%/%%a_attention_window_%%w
        
        REM Execute the training command
        echo Executing: python main.py --mode train --epochs %BASE_EPOCHS% --batch_size %BASE_BATCH_SIZE% --lr %BASE_LR% --input_window %BASE_INPUT_WINDOW% --output_window %%w --num_channels %BASE_NUM_CHANNELS% --kernel_size %BASE_KERNEL_SIZE% --num_runs %BASE_NUM_RUNS% --model_dir !current_model_dir! --wandb_project %BASE_WANDB_PROJECT% %ADDITIONAL_FLAGS% --attention_type %%a --chunk_size %BASE_CHUNK_SIZE%
        echo.
        
        python main.py --mode train --epochs %BASE_EPOCHS% --batch_size %BASE_BATCH_SIZE% --lr %BASE_LR% --input_window %BASE_INPUT_WINDOW% --output_window %%w --num_channels %BASE_NUM_CHANNELS% --kernel_size %BASE_KERNEL_SIZE% --num_runs %BASE_NUM_RUNS% --model_dir !current_model_dir! --wandb_project %BASE_WANDB_PROJECT% %ADDITIONAL_FLAGS% --attention_type %%a --chunk_size %BASE_CHUNK_SIZE%
        
        REM Record end time and calculate duration
        set run_end_time=!time!
        call :time_to_seconds !run_end_time! run_end_seconds
        set /a run_duration=!run_end_seconds!-!run_start_seconds!
        
        REM Handle negative duration (crossing midnight)
        if !run_duration! LSS 0 set /a run_duration=!run_duration!+86400
        
        REM Update running average of execution time
        if !run_counter! EQU 1 (
            set /a estimated_time_per_run=!run_duration!
        ) else (
            set /a prev_runs=!run_counter!-1
            set /a temp_calc=!estimated_time_per_run!*!prev_runs!
            set /a temp_calc=!temp_calc!+!run_duration!
            set /a estimated_time_per_run=!temp_calc!/!run_counter!
        )
        
        REM Convert duration to readable format
        call :seconds_to_time !run_duration! run_duration_str
        call :seconds_to_time !estimated_time_per_run! avg_duration_str
        
        REM Check if the command was successful
        if !ERRORLEVEL! NEQ 0 (
            echo ERROR: Training failed for attention_type=%%a, output_window=%%w
            echo Error code: !ERRORLEVEL!
            echo Duration: !run_duration_str!
            echo.
        ) else (
            echo SUCCESS: Training completed for attention_type=%%a, output_window=%%w
            echo Duration: !run_duration_str!
            echo Average duration per run: !avg_duration_str!
            echo.
        )
        
        REM Optional: Add a small delay between runs
        timeout /t 2 /nobreak >nul
    )
)

REM Calculate total execution time
set total_end_time=%time%
call :time_to_seconds %total_end_time% total_end_seconds
set /a total_duration=!total_end_seconds!-!total_start_seconds!
if !total_duration! LSS 0 set /a total_duration=!total_duration!+86400
call :seconds_to_time !total_duration! total_duration_str

echo.
echo =====================================================
echo All training configurations completed!
echo Total runs executed: !run_counter!
echo Total execution time: !total_duration_str!
echo Average time per configuration: !avg_duration_str!
echo Start time: %total_start_time%
echo End time: %total_end_time%
echo =====================================================
echo.

REM Generate detailed summary report
echo Generating detailed summary report...
echo Training Summary Report > training_summary.txt
echo ======================= >> training_summary.txt
echo Date: %date% %time% >> training_summary.txt
echo Total configurations run: !run_counter! >> training_summary.txt
echo Total execution time: !total_duration_str! >> training_summary.txt
echo Average time per configuration: !avg_duration_str! >> training_summary.txt
echo Start time: %total_start_time% >> training_summary.txt
echo End time: %total_end_time% >> training_summary.txt
echo. >> training_summary.txt
echo Configurations: >> training_summary.txt
for %%a in (%attention_types%) do (
    for %%w in (%output_windows%) do (
        echo - attention_type: %%a, output_window: %%w, model_dir: %BASE_MODEL_DIR%/%%a_attention_window_%%w >> training_summary.txt
    )
)

echo Summary report saved to training_summary.txt
echo.
echo Press any key to exit...
pause >nul
goto :eof

REM =====================================================
REM Helper Functions for Time Calculations
REM =====================================================

:time_to_seconds
REM Convert HH:MM:SS.ss to total seconds
set time_str=%1
for /f "tokens=1-4 delims=:." %%a in ("%time_str%") do (
    set /a hours=%%a
    set /a minutes=%%b
    set /a seconds=%%c
)
REM Remove leading zeros to avoid octal interpretation
set /a hours=1%hours%-100
set /a minutes=1%minutes%-100
set /a seconds=1%seconds%-100
set /a %2=hours*3600+minutes*60+seconds
goto :eof

:seconds_to_time
REM Convert seconds to HH:MM:SS format
set /a total_sec=%1
set /a hours=!total_sec!/3600
set /a temp_remainder=!total_sec! %% 3600
set /a minutes=!temp_remainder!/60
set /a seconds=!total_sec! %% 60
if !hours! LSS 10 set hours=0!hours!
if !minutes! LSS 10 set minutes=0!minutes!
if !seconds! LSS 10 set seconds=0!seconds!
set %2=!hours!:!minutes!:!seconds!
goto :eof

:calculate_eta
REM Calculate estimated time of completion
set /a current_run=%1
set /a total_runs=%2
set /a avg_time=%3
set /a remaining_runs=!total_runs!-!current_run!
set /a remaining_runs=!remaining_runs!+1
set /a eta_seconds=!remaining_runs!*!avg_time!

call :seconds_to_time !eta_seconds! eta_time_str
echo Estimated time remaining: !eta_time_str!

REM Calculate estimated completion time
call :time_to_seconds !time! current_seconds
set /a completion_seconds=!current_seconds!+!eta_seconds!
if !completion_seconds! GEQ 86400 set /a completion_seconds=!completion_seconds!-86400
call :seconds_to_time !completion_seconds! completion_time_str
echo Estimated completion time: !completion_time_str!
goto :eof

:error_exit
echo.
echo =====================================================
echo Script terminated due to error
echo =====================================================
pause >nul
exit /b 1