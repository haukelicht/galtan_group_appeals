from datetime import datetime
ts = lambda: datetime.now().replace(microsecond=0).isoformat().replace('T', ' ')
log = lambda *args, **kwargs: print(f'[{ts()}]:', *args, **kwargs)
