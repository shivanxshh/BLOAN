mkdir -p ~/.streamlit
echo "[server]" >> ~/.streamlit/config.toml   ##configuring port on server
echo "port = $PORT" >> ~/.streamlit/config.toml