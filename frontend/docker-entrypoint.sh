#!/bin/sh

# Replace environment variables in JavaScript files
for file in /usr/share/nginx/html/assets/*.js; do
  if [ -f "$file" ]; then
    sed -i 's|VITE_API_URL_PLACEHOLDER|'${VITE_API_URL}'|g' "$file"
  fi
done

exec "$@"
