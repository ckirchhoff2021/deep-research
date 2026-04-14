#!/bin/bash

echo "Stopping Streamlit apps..."

get_streamlit_ports_for_pid() {
    local pid="$1"

    if command -v lsof >/dev/null 2>&1; then
        lsof -Pan -p "$pid" -iTCP -sTCP:LISTEN 2>/dev/null \
            | awk 'NR > 1 { split($9, parts, ":"); print parts[length(parts)] }' \
            | sort -u
        return
    fi

    if command -v ss >/dev/null 2>&1; then
        ss -ltnp 2>/dev/null \
            | awk -v pid="$pid" '$0 ~ ("pid=" pid ",") { split($4, parts, ":"); print parts[length(parts)] }' \
            | sort -u
    fi
}

kill_listening_pids_by_port() {
    local port="$1"
    local port_pids=""

    if command -v lsof >/dev/null 2>&1; then
        port_pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null)
    elif command -v ss >/dev/null 2>&1; then
        port_pids=$(ss -ltnp 2>/dev/null | awk -v port="$port" '$4 ~ (":" port "$") { if (match($0, /pid=[0-9]+/)) { print substr($0, RSTART + 4, RLENGTH - 4) } }' | sort -u)
    fi

    if [ -z "$port_pids" ]; then
        return
    fi

    for port_pid in $port_pids; do
        kill -9 "$port_pid" 2>/dev/null && echo "Killed process $port_pid on port $port"
    done
}

streamlit_pids=$(pgrep -f "streamlit run")

if [ -z "$streamlit_pids" ]; then
    echo "No running Streamlit app found."
    exit 0
fi

ports=()

for pid in $streamlit_pids; do
    while IFS= read -r port; do
        [ -z "$port" ] && continue

        already_added=false
        for existing_port in "${ports[@]}"; do
            if [ "$existing_port" = "$port" ]; then
                already_added=true
                break
            fi
        done

        if [ "$already_added" = false ]; then
            ports+=("$port")
        fi
    done < <(get_streamlit_ports_for_pid "$pid")

    kill "$pid" 2>/dev/null && echo "Sent SIGTERM to Streamlit process $pid"
done

sleep 2

for pid in $streamlit_pids; do
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null && echo "Force killed Streamlit process $pid"
    fi
done

for port in "${ports[@]}"; do
    kill_listening_pids_by_port "$port"
done

echo "Streamlit apps stopped."

