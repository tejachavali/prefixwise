# The CHANGELOG section for version v (`## v` or `## v: date`), without its heading: release notes.
BEGIN { h = "## " v }
index($0, h) == 1 && (length($0) == length(h) || substr($0, length(h) + 1, 1) ~ /[: ]/) { on = 1; next }
on && /^## / { exit }
on { print }
