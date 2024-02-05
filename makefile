build:
	docker compose build --no-cache

start:
	docker compose up

restart:
	docker compose rm -v -f
	docker compose up