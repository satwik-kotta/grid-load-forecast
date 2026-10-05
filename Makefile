.PHONY: etl forecast causal dashboard test all
etl:       ; python -W ignore -m src.etl
forecast:  ; python -W ignore -m src.forecast
causal:    ; python -W ignore -m src.causal
dashboard: ; python -W ignore -m src.export_dashboard
test:      ; python -W ignore -m pytest -q tests
all: etl forecast causal dashboard
