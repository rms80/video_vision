# video_vision



## Dev server

To start or restart the Vite dev server, run `bash scripts/restart-dev.sh`.
It frees port 4444 (kills any existing listener) and runs `npm run dev`.
Don't `npm run dev` directly — `strictPort: true` means a stale listener
on 4444 will fail the start.
