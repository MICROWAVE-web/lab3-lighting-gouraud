#!/bin/bash
cd "$(dirname "$0")"
pick=""
for py in /usr/local/bin/python3 /usr/bin/python3 python3; do
  if [ -x "$py" ] || command -v "$py" >/dev/null 2>&1; then
    if "$py" -c "import tkinter" 2>/dev/null; then
      pick="$py"
      break
    fi
  fi
done
if [ -z "$pick" ]; then
  echo "Не найден Python с tkinter."
  exit 1
fi
echo "Запуск: $pick"
exec "$pick" main.py
