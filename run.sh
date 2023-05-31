OLD_PID=$(lsof -t -i:8080)
echo "OLD_PID: $OLD_PID"
kill $OLD_PID
sleep 1
poetry run python infer-web.py
