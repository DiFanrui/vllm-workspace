#!/usr/bin/env bash
set -u

worktree=/root/autodl-tmp/InfiniLM-dpv4-test
killed=0

for proc_dir in /proc/[0-9]*; do
    comm=$(cat "$proc_dir/comm" 2>/dev/null || true)
    cwd=$(readlink "$proc_dir/cwd" 2>/dev/null || true)
    if [[ "$comm" == "cc1plus" && "$cwd" == "$worktree" ]]; then
        kill "${proc_dir##*/}"
        killed=$((killed + 1))
    fi
done

for pid in 7050 7049; do
    if [[ -d "/proc/$pid" ]]; then
        kill "$pid"
        killed=$((killed + 1))
    fi
done

echo "terminated=$killed"
