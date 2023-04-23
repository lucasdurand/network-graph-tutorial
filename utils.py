from operator import itemgetter
def top_n(d: dict[str, int], *, n:int=1, break_ties:bool=False) -> dict[str, int]:
    """Return the top ``n`` entries, optionally breaking ties"""
    ranked = sorted(d.items(), key=itemgetter(1), reverse=True)
    top = {k:v for k,v in ranked[:n]}
    if break_ties: # arbitrarily choose winners
        return top
    else: # include all qualifiers
        good_enough = set(top.values())
        return {k:v for k,v in ranked if v in good_enough} 