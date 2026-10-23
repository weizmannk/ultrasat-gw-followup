#!/usr/bin/env bash
# Swap the size-limited PyPI `cplex` runtime for the full CPLEX Optimization
# Studio engine. Run this once after installing CPLEX Studio, and again after
# any reinstall of the `cplex` wheel (`uv sync`, `uv pip install`, ...).
#
#   ./scripts/setup-cplex.sh [/path/to/CPLEX_Studio]
#
# With no argument the usual install locations are probed:
#   Linux  /opt/ibm/ILOG/CPLEX_Studio*   ~/opt/ibm/ILOG/CPLEX_Studio*
#          ~/CPLEX_Studio*
#   macOS  ~/Applications/CPLEX_Studio*  /Applications/CPLEX_Studio*
#
set -euo pipefail

find_studio() {
    [ $# -gt 0 ] && { printf '%s\n' "$1"; return; }
    for base in \
        "${CPLEX_STUDIO_DIR:-}" \
        /opt/ibm/ILOG/CPLEX_Studio* \
        "$HOME"/opt/ibm/ILOG/CPLEX_Studio* \
        "$HOME"/CPLEX_Studio* \
        "$HOME"/Applications/CPLEX_Studio* \
        /Applications/CPLEX_Studio*
    do
        [ -n "$base" ] && [ -d "$base/cplex" ] && { printf '%s\n' "$base"; return; }
    done
}

STUDIO="$(find_studio "$@")"

if [ -z "$STUDIO" ] || [ ! -d "$STUDIO/cplex" ]; then
    echo "CPLEX Studio not found." >&2
    echo "Pass the install path explicitly, e.g.:" >&2
    echo "  $0 /opt/ibm/ILOG/CPLEX_Studio222" >&2
    exit 1
fi

echo "Using CPLEX Studio: $STUDIO"
docplex config --upgrade "$STUDIO"

python - <<'PY'
import cplex

c = cplex.Cplex()
c.variables.add(names=[f"x{i}" for i in range(3000)])
c.objective.set_linear([(f"x{i}", 1.0) for i in range(3000)])
c.solve()
print(f"OK: full CPLEX {cplex.__version__} active (no 1000-variable limit)")
PY
