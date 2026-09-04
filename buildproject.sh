#!/usr/bin/env bash
# clarautils installieren/aktualisieren (editable, pip install -e .)
# Ziel-venv: die aktivierte venv, sonst die Repo-.venv.
# Extra pip-Args werden durchgereicht.
# Beispiel: ./buildproject.sh --upgrade

set -euo pipefail

cd "$(dirname "$(realpath "$0")")"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
else
    PY=".venv/bin/python"
fi

if [[ ! -x "$PY" ]]; then
    echo "Python in der virtuellen Umgebung nicht gefunden: $PY" >&2
    echo "Aktiviere eine venv oder erstelle im Repository eine mit:" >&2
    echo "  python -m venv .venv" >&2
    exit 1
fi

echo "Installing clarautils (editable) with: $PY"
"$PY" -m pip install -e . "$@"

"$PY" -c "import clarautils; from clarautils import Bitty, DefinedBit; print('clarautils', clarautils.__version__, 'OK')"