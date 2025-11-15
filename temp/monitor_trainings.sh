#!/bin/bash

JOBS="1639625 1639628 1639629 1639630"
LOG_DIR="/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/logs"
CHECK_INTERVAL=1200  # 20 minutes

echo "=== ML Training Monitor Started at $(date) ==="
echo "Monitoring jobs: $JOBS"
echo "Check interval: ${CHECK_INTERVAL}s (20 minutes)"
echo ""

# Initial sleep
echo "Initial sleep for 20 minutes..."
sleep $CHECK_INTERVAL

while true; do
    echo ""
    echo "=== Status Check at $(date) ==="
    
    # Check if any jobs are still in queue
    RUNNING=$(condor_q $JOBS -nobatch 2>/dev/null | grep -E "^[0-9]+" | wc -l)
    
    if [ "$RUNNING" -eq 0 ]; then
        echo "All jobs completed or removed from queue."
        
        # Check for failures
        echo ""
        echo "=== Checking for failures ==="
        
        for JOB in $JOBS; do
            EXIT_CODE=$(condor_history $JOB -limit 1 -af ExitCode 2>/dev/null)
            JOB_STATUS=$(condor_history $JOB -limit 1 -af JobStatus 2>/dev/null)
            
            if [ "$EXIT_CODE" != "0" ] || [ "$JOB_STATUS" != "4" ]; then
                echo "Job $JOB failed (ExitCode: $EXIT_CODE, Status: $JOB_STATUS)"
                
                # Find the corresponding log files
                ERROR_LOG=$(ls ${LOG_DIR}/*${JOB}*.err 2>/dev/null | head -1)
                OUTPUT_LOG=$(ls ${LOG_DIR}/*${JOB}*.out 2>/dev/null | head -1)
                
                if [ -f "$ERROR_LOG" ]; then
                    echo "  Error log: $ERROR_LOG"
                    echo "  Last 30 lines:"
                    tail -30 "$ERROR_LOG" | sed 's/^/    /'
                fi
            else
                echo "Job $JOB completed successfully"
            fi
        done
        
        echo ""
        echo "=== Monitoring complete at $(date) ==="
        break
    else
        echo "Jobs still running: $RUNNING"
        condor_q $JOBS -nobatch 2>/dev/null | head -20
        
        # Check for held jobs
        HELD=$(condor_q $JOBS -nobatch 2>/dev/null | grep " H " | wc -l)
        if [ "$HELD" -gt 0 ]; then
            echo ""
            echo "WARNING: $HELD jobs are held!"
            condor_q $JOBS -nobatch -af ClusterId HoldReason 2>/dev/null
        fi
        
        echo ""
        echo "Next check in ${CHECK_INTERVAL}s..."
        sleep $CHECK_INTERVAL
    fi
done
