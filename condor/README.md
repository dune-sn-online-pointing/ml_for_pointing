# HTCondor Submit Scripts

This directory contains HTCondor submit scripts organized by network type.

## Structure

- `channel_tagging/` - Channel tagging network submissions
  - `submit_production_v4_100k.sub` - 100k sample training
  - `submit_production_v4_full.sub` - Full dataset training
  - `submit_production_v4_streaming.sub` - Streaming dataset training
  
- `mt_identifier/` - Main track identifier submissions
  - `submit_production_v5.sub` - Production v5 training
  
- `electron_direction/` - Electron direction regression submissions
  - `submit_production_v1.sub` - Production v1 training

## Submitting Jobs

From the repository root:
```bash
condor_submit condor/<network_type>/submit_production_<version>.sub
```

## Configurations

JSON configurations are in `json/<network_type>/production_<version>.json`
