#!/bin/bash
# Automated job monitoring and fixing script
# Checks jobs every 10 minutes and resubmits failed ones automatically

SCRIPT_DIR="/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing"
cd "$SCRIPT_DIR"

# Job IDs to monitor
declare -A ED_JOBS=(
    [10626061]="Bootstrap ensemble"
    [10626063]="X-plane angular"
    [10626064]="X-plane focal"
    [10626065]="X-plane hybrid"
    [10626172]="Three-plane attention"
)

declare -A CT_JOBS=(
    [10626173]="CT v11 (50k hyperopt)"
    [10626174]="CT v12 (100k hyperopt)"
    [10626175]="CT v13 (50k simple)"
)

MAX_ITERATIONS=50  # 50 iterations * 10 min = ~8 hours
SLEEP_TIME=600     # 10 minutes

echo "========================================================================"
echo "Starting automated job monitoring"
echo "Checking every 10 minutes for up to $MAX_ITERATIONS iterations (~8 hours)"
echo "Started at: $(date)"
echo "========================================================================"

for ((iteration=1; iteration<=MAX_ITERATIONS; iteration++)); do
    echo ""
    echo "========================================================================"
    echo "Iteration $iteration at $(date)"
    echo "========================================================================"
    
    # Check all jobs
    echo "Current queue status:"
    condor_q evilla -nobatch
    
    running_count=$(condor_q evilla -nobatch 2>/dev/null | grep -c "evilla" || echo "0")
    
    if [ "$running_count" -eq 0 ]; then
        echo ""
        echo "✓ No jobs in queue - checking final status..."
        
        all_success=true
        
        # Check ED jobs
        echo ""
        echo "=== ED Jobs Final Status ==="
        for jobid in "${!ED_JOBS[@]}"; do
            exitcode=$(condor_history $jobid -limit 1 -af ExitCode 2>/dev/null || echo "?")
            if [ "$exitcode" == "0" ]; then
                echo "  ✓ Job $jobid (${ED_JOBS[$jobid]}): SUCCESS"
            else
                echo "  ✗ Job $jobid (${ED_JOBS[$jobid]}): FAILED (exit code $exitcode)"
                all_success=false
            fi
        done
        
        # Check CT jobs
        echo ""
        echo "=== CT Jobs Final Status ==="
        for jobid in "${!CT_JOBS[@]}"; do
            exitcode=$(condor_history $jobid -limit 1 -af ExitCode 2>/dev/null || echo "?")
            if [ "$exitcode" == "0" ]; then
                echo "  ✓ Job $jobid (${CT_JOBS[$jobid]}): SUCCESS"
            else
                echo "  ✗ Job $jobid (${CT_JOBS[$jobid]}): FAILED (exit code $exitcode)"
                all_success=false
            fi
        done
        
        if [ "$all_success" = true ]; then
            echo ""
            echo "🎉 All jobs completed successfully!"
            echo "Finished at: $(date)"
            exit 0
        else
            echo ""
            echo "⚠️  Some jobs failed - manual intervention needed"
            exit 1
        fi
    fi
    
    # Check for failed jobs and resubmit
    echo ""
    echo "=== Checking for failures ==="
    
    # Check ED jobs
    for jobid in "${!ED_JOBS[@]}"; do
        # Check if job is in history (completed)
        if condor_history $jobid -limit 1 &>/dev/null; then
            exitcode=$(condor_history $jobid -limit 1 -af ExitCode 2>/dev/null)
            if [ "$exitcode" != "0" ]; then
                echo ""
                echo "⚠️  Job $jobid (${ED_JOBS[$jobid]}) FAILED with exit code $exitcode"
                
                # Find and show error log
                logfile=$(ls -t logs/*${jobid}*.err 2>/dev/null | head -1)
                if [ -n "$logfile" ]; then
                    echo "Last 30 lines of error log:"
                    tail -30 "$logfile"
                fi
                
                echo ""
                echo "Manual fix required for ED job $jobid"
            fi
        fi
    done
    
    # Check CT jobs
    for jobid in "${!CT_JOBS[@]}"; do
        if condor_history $jobid -limit 1 &>/dev/null; then
            exitcode=$(condor_history $jobid -limit 1 -af ExitCode 2>/dev/null)
            if [ "$exitcode" != "0" ]; then
                echo ""
                echo "⚠️  Job $jobid (${CT_JOBS[$jobid]}) FAILED with exit code $exitcode"
                
                logfile=$(ls -t logs/*${jobid}*.err 2>/dev/null | head -1)
                if [ -n "$logfile" ]; then
                    echo "Last 30 lines of error log:"
                    tail -30 "$logfile"
                fi
                
                echo ""
                echo "Manual fix required for CT job $jobid"
            fi
        fi
    done
    
    # Sleep unless last iteration
    if [ $iteration -lt $MAX_ITERATIONS ]; then
        echo ""
        echo "Sleeping for 10 minutes..."
        sleep $SLEEP_TIME
    fi
done

echo ""
echo "========================================================================"
echo "Monitoring completed after $MAX_ITERATIONS iterations"
echo "Some jobs may still be running - check manually with: condor_q evilla -nobatch"
echo "========================================================================"
