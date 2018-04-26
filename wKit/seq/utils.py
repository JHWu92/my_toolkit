
def similarity(a,b):
    import difflib
    from constants import no_name_value
    if len(a)==0 or len(b)==0:
        return no_name_value
    a,b = make_type_consistent(a,b)
    a,b = a.lower(), b.lower()
    match_size = sum([m.size for m in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks()])*1.0
    longer_size, shorter_size = (len(b),len(a)) if len(a)<len(b) else ((len(a),len(b)))
    longer_size = shorter_size if (longer_size/1.5)<=shorter_size else longer_size/1.5
#     return match_size/(longer_size+shorter_size)*2
    return match_size/shorter_size