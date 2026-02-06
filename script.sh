#!/usr/bin/env bash
set -euo pipefail

ROOT="."

# Find empty files and format as a tree-like structure
find "$ROOT" -type f -empty | sort | awk '
BEGIN {
    print "Empty files structure:"
}
{
    # remove leading "./"
    sub(/^\.\//, "", $0)

    n = split($0, parts, "/")
    indent = ""

    for (i = 1; i <= n; i++) {
        if (i == n) {
            printf "%s└── %s\n", indent, parts[i]
        } else {
            printf "%s├── %s/\n", indent, parts[i]
            indent = indent "│   "
        }
    }
}
'
