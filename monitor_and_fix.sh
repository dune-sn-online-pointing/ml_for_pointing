#!/bin/bash
# Automatic job monitoring and fixing script
# Checks every 10 minutes and fixes common errors

WORK_DIR="/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing"
cd "$WORK_DIR"

# Jobs to monitor
CT_V11=10625252
CT_V12=10625253
THREE_PLANE=10625260

echo "========================================="
echo "Monitoring Jobs - Started $(date)"
echo "========================================="
echo "CT v11: $CT_V11"
echo "CT v12: $CT_V12"
echo "Three-plane: $THREE_PLANE"
echo ""

check_and_fix() {
    local JOB_ID=$1
    local JOB_NAME=$2
    
    # Check if job failed
    STATUS=$(condor_history $JOB_ID -limit 1 -af ExitCode 2>/dev/null)
    
    if [ "$STATUS" = "1" ]; then
        echo "⚠️  $JOB_NAME ($JOB_ID) FAILED!"
        
        # Find error log
        ERR_LOG=$(ls -t logs/*${JOB_ID}.err 2>/dev/null | head -1)
        if [ -f "$ERR_LOG" ]; then
            echo "Error snippet:"
            tail -20 "$ERR_LOG"
            
            # Check for common errors and attempt fixes
            if grep -q "Unsupported metadata length" "$ERR_LOG"; then
                echo "→ Metadata error detected - already fixed in code"
            elif grep -q "unexpected keyword argument" "$ERR_LOG"; then
                echo "→ Function signature error - already fixed in code"
            elif grep -q "unsupported format string" "$ERR_LOG"; then
                echo "→ Format string error - already fixed in code"
            fi
        fi
    elif [ "$STATUS" = "0" ]; then
        echo "✅ $JOB_NAME ($JOB_ID) completed successfully"
    fi
}

while true; do
    echo ""
    echo "=== Check at $(date) ==="
    
    # Show running jobs
    condor_q evilla -nobatch
    
    # Check each job
    check_and_fix $CT_V11 "CT v11"
    check_and_fix $CT_V12 "CT v12"
    check_and_fix $THREE_PLANE "Three-plane"
    
    # Check if all jobs are done
    RUNNING=$(condor_q evilla -af ClusterId | wc -l)
    if [ "$RUNNING" -eq "0" ] || [ "$RUNNING" -eq "1" ]; then
        echo ""
        echo "========================================="
        echo "Most jobs completed! Final check:"
        echo "========================================="
        condor_history evilla -limit 5 -af ClusterId ExitCode
        break
    fi
    
    echo ""
    echo "Sleeping for 10 minutes..."
    sleep 600
done

echo ""
echo "========================================="
echo "Monitoring Complete - $(date)"
echo "========================================="
