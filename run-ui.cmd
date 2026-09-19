@echo off
cd /d "%~dp0..\astra-ui-app"
set DEV_AUTH_BYPASS=1
set FLASK_SECRET=dev-secret
set USE_ENGINE_HTTP=1
set RADIO_ENGINE_URL=http://localhost:8010
set CONSTELLATION_ENGINE_URL=http://localhost:8020
set PYTHONPATH=%~dp0
python satellite_planner.py
