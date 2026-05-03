def binance_worker(radar_obj):
    radar_obj.get_working_rest_url()
    while True:
        try:
            url = f"{radar_obj.rest_base_url}/fapi/v1/ticker/24hr"
            r = requests.get(url, timeout=5)
            print(f"[WORKER] Status: {r.status_code}, Pairs: {len(r.json())}")
            if r.status_code == 200:
                raw = r.json()
                formatted = [{'s': x['symbol'], 'c': x['lastPrice'], 'q': x['quoteVolume']} for x in raw]
                radar_obj.process_ticker(formatted)
                print(f"[WORKER] Signals: {len(radar_obj.signals)}, History: {len(radar_obj.history)}")
        except Exception as e:
            print(f"[WORKER ERROR] {e}")
        time.sleep(FETCH_INTERVAL)
