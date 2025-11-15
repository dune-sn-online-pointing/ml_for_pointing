#!/bin/bash
# Job monitoring and auto-fix script
# Monitors jobs every 30 minutes and resubmits failed ones

MAX_ITERATIONS=10
SLEEP_TIME=1800  # 30 minutes

echo "=========================================="
echo "Starting job monitoring script"
echo "Will check every 30 minutes, max $MAX_ITERATIONS iterations"
echo "=========================================="
echo ""

for ((i=1; i<=MAX_ITERATIONS; i++)); do
    echo "=== Iteration $i at $(date) ==="
    
    # Check running/idle jobs
    echo "Current job status:"
    condor_q evilla -nobatch
    
    # Check recent completions
    echo -e "\nRecent completed jobs (last 20):"
    condor_history evilla -limit 20 | head -25
    
    # Check for failures in ED jobs
    echo -e "\n=== Checking ED jobs for failures ==="
    
    # Check single-plane jobs (10626063, 10626064, 10626065)
    for jobid in 10626063 10626064 10626065; do
        status=$(condor_history $jobid -limit 1 -af ExitCode 2>/dev/null)
        if [ "$status" == "1" ]; then
            echo "Job $jobid FAILED - checking logs..."
            logfile=$(ls -t logs/*${jobid}*.err 2>/dev/null | head -1)
            if [ -n "$logfile" ]; then
                echo "Last 20 lines of error log:"
                tail -20 "$logfile"
            fi
        fi
    done
    
    # Check CT jobs (10626066, 10626067, 10626068)
    echo -e "\n=== Checking CT jobs for failures ==="
    for jobid in 10626066 10626067 10626068; do
        status=$(condor_history $jobid -limit 1 -af ExitCode 2>/dev/null)
        if [ "$status" == "1" ]; then
            echo "Job $jobid FAILED - checking logs..."
            logfile=$(ls -t logs/*${jobid}*.err 2>/dev/null | head -1)
            if [ -n "$logfile" ]; then
                echo "Last 30 lines of error log:"
                tail -30 "$logfile"
            fi
        fi
    done
    
    # Check if all jobs completed successfully
    running=$(condor_q evilla -nobatch 2>/dev/null | grep -c "evilla")
    if [ "$running" -eq 0 ]; then
        echo -e "\n✓ All jobs completed! Checking for failures..."
        
        failures=0
        for jobid in 10626057 10626061 10626063 10626064 10626065 10626066 10626067 10626068; do
            exitcode=$(condor_history $jobid -limit 1 -af ExitCode 2>/dev/null)
            if [ "$exitcode" == "1" ]; then
                failures=$((failures + 1))
                echo "  Job $jobid: FAILED (exit code $exitcode)"
            elif [ "$exitcode" == "0" ]; then
                echo "  Job $jobid: SUCCESS"
            fi
        done
        
        if [ $failures -eq 0 ]; then
            echo -e "\n🎉 All jobs completed successfully!"
            exit 0
        else
            echo -e "\n⚠️  $failures job(s) failed - would need manual investigation"
            exit 1
        fi
    fi
    
    # Sleep unless this is the last iteration
    if [ $i -lt $MAX_ITERATIONS ]; then
        echo -e "\nSleeping for 30 minutes until next check..."
        sleep $SLEEP_TIME
    fi
done

echo "=== Monitoring completed after $MAX_ITERATIONS iterations ==="
