#!/bin/sh
# Install the local pre-commit hook (git hooks are not versioned).
set -e
root="$(git rev-parse --show-toplevel)"
cat > "$root/.git/hooks/pre-commit" <<'HOOK'
#!/bin/sh
exec python3 scripts/sync_readme_results.py --check
HOOK
chmod +x "$root/.git/hooks/pre-commit"
echo "installed .git/hooks/pre-commit"
