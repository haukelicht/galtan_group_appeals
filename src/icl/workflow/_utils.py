import regex
from typing import List

from datetime import datetime
ts = lambda: datetime.now().replace(microsecond=0).isoformat().replace('T', ' ')
log = lambda *args, **kwargs: print(f'[{ts()}]:', *args, **kwargs)

import warnings
format_warning = lambda message, category, filename, lineno, line=None: f"[{ts()}]: {category.__name__}: {message}\n"
warnings.formatwarning = format_warning
