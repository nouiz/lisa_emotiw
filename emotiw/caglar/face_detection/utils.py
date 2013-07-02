

def list_alternator(lst1, lst2):
    """
    A basic function to alternate and merge two different lists.
    """
    sz = min(len(lst1), len(lst2))
    altenated = [None] * 2 * sz
    #First fill in the odd valued indices:
    alternated[0::2] = lst1
    #Then fill in the even valued indices:
    alternated[1::2] = lst2
    alternated.extend(lst1)
    alternated.extend(lst2)
    return alternated

