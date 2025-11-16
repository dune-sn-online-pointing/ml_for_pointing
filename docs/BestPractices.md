Remember:
- Instead of directing unwanted output to /dev/null, use the proper flags for the different commands to silence output
- for each task having their subfolder (electron_direction, mt_identifier and channel_tagging) follow the structure to  place sub scripts under condor/, logs are supposed to be under logs/, analysis tools under ana/.
- ask for a GPU when submitting trainings
- output of trainings shall go in /eos/user/e/evilla/dune/sn-tps/neural_networks/ in the appropriate subfolder.
- test the trainings with a json with reduced dataset in order to verify a configuration works before submitting the job.
- always use progressive version numbers for  models to avoid confusion.
- update Networks.md with info about new models, no need to add if it's running or what.
- source in the current shell source scripts/init.sh to load python env