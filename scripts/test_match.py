def fuzzy_keyword_match(word, map_dict, threshold=None):
    if not word or not map_dict:
        return None, None
    clean_w = word.upper().strip('.,!?;:"\'-')
    if not clean_w:
        return None, None
    if clean_w in map_dict:
        return clean_w, map_dict[clean_w]
    for k, v in map_dict.items():
        if clean_w.startswith(k) or clean_w.endswith(k):
            if len(k) >= 3:
                return k, v
    return None, None

EMOJI_MAP = {
    'DUIT': '[$$]', 'CASH': '[$$]', 'CUAN': '[$$]', 'HATI': '<3',
    'SETUJU': '[OK]', 'SALAH': '[X]', 'API': '*', 'NAGA': '>>',
    'SAHAM': '[+]', 'BETOT': '>>', 'BEARISH': '[-]', 'BULLISH': '[+]',
}
SFX_MAP = {
    'BOOM': 'vine_boom.mp3', 'WOW': 'ding.mp3', 'BAGUS': 'ding.mp3',
    'KEREN': 'ding.mp3', 'TIPS': 'pop.mp3', 'RAHASIA': 'pop.mp3',
    'HEY': 'whoosh.mp3', 'TRANSISI': 'whoosh.mp3', 'CUAN': 'ding.mp3',
    'BETOT': 'whoosh.mp3', 'HAJAR': 'vine_boom.mp3',
}

tests = ['BERISNYA', 'DIBETOT', 'BEARISH', 'BULLISH', 'PROFITNYA', 'CUAN', 'TERUS', 'KAYAK', 'NAGANYA']
print('=== EMOJI ===')
for w in tests:
    k, v = fuzzy_keyword_match(w, EMOJI_MAP)
    print(f'  {w:16s} => {k} -> {v}' if k else f'  {w:16s} => NO MATCH')
print('=== SFX ===')
for w in tests:
    k, v = fuzzy_keyword_match(w, SFX_MAP)
    print(f'  {w:16s} => {k} -> {v}' if k else f'  {w:16s} => NO MATCH')
