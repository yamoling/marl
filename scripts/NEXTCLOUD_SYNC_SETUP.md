# Nextcloud Sync Setup Guide

This guide explains how to use the Nextcloud synchronization for your logs directory.

## Quick Start

1. **Install dependencies** (already declared in `pyproject.toml`):
   ```bash
   uv sync
   ```

2. **Run the sync script:**
   ```bash
   python scripts/sync_nextcloud.py
   ```

The script reads your credentials from `.env` and syncs your logs.

## Configuration

Credentials are stored in `.env` at the repo root:

```bash
NEXTCLOUD_URL="https://mycloud.ulb.be"
NEXTCLOUD_USERNAME="your_username"
NEXTCLOUD_PASSWORD="your_app_password"
```

Use an app-specific password (Nextcloud settings → Security → Devices & sessions), not your main account password.

⚠️ **Important:** `.env` is gitignored and will NOT be committed to git.

## Usage Examples

```bash
# Bidirectional sync (default)
python scripts/sync_nextcloud.py

# Upload only (push local logs to Nextcloud)
python scripts/sync_nextcloud.py --direction up

# Download only (pull from Nextcloud to local)
python scripts/sync_nextcloud.py --direction down

# Use custom local directory
python scripts/sync_nextcloud.py --local my_logs

# Use custom remote directory
python scripts/sync_nextcloud.py --remote MyLogs/experiments

# Tune upload/download concurrency (default: 8)
python scripts/sync_nextcloud.py --workers 16

# Check whether the remote directory exists (and create it if missing)
python scripts/sync_nextcloud.py --check-remote
```

## How It Works

- **WebDAV Protocol:** Uses WebDAV for compatibility with Nextcloud.
- **Change Detection:** Compares mtime+size (like rsync's default quick check) rather than hashing file content — avoids reading every file twice (once to check, once to transfer).
- **`test` Directories Excluded:** Any directory (or file) named `test` is pruned from the scan entirely and never synced in either direction.
- **Parallel Transfers:** Uploads/downloads run concurrently across a thread pool (`--workers`), each with its own WebDAV client.
- **Bulk Directory Creation:** Remote directories are created level-by-level, in parallel, rather than one blocking request per file — important for trees with many nested experiment folders.
- **Smart Caching:** Uses `.sync_cache.json` to skip files that haven't changed since the last sync.
- **Fail-Fast:** Exits immediately on critical errors (auth failure, missing directories, a failed transfer).

## Troubleshooting

### "webdav3 library not installed" / "No module named 'dotenv'"
```bash
uv sync
```

### Authentication fails
1. Verify `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME`, `NEXTCLOUD_PASSWORD` in `.env`
2. Regenerate the app-specific password if unsure
3. Check that the Nextcloud server is accessible

### Files not syncing
1. Check that both local and remote directories exist
2. Verify you have read/write permissions on Nextcloud
3. Check `.sync_cache.json` to see what was last synced
4. Run with `--direction up` or `--direction down` to diagnose

### It looks stuck / no output
For very large trees, the initial scan (and first directory-creation pass) can take a little while — progress prints appear every 20,000 files scanned and every 500 directories created. If you see genuinely nothing for a long time, check that stdout isn't being swallowed by whatever is running the script.

## Automation

You can automate syncs using cron:

```bash
# Edit crontab
crontab -e

# Add a line to sync every hour (example):
0 * * * * cd /workspaces/marl && python scripts/sync_nextcloud.py >> logs/sync.log 2>&1
```

Or create a systemd timer for more control.

## Security Notes

- ✅ Credentials stored locally only, in `.env` (not in git)
- ✅ Uses HTTPS for secure communication
- ✅ App-specific password (not your main Nextcloud password)
- ✅ Local cache file is gitignored
