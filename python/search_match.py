import re

match_ = re.search("bc", "abcd")
if match_:
    print(match_.group()) # whole match is returned
    print(match_[0])
    print(match_.groups()) # returns subgroups
    print(match_.groupdict()) # returns named subgroups
    """
    bc : group()
    bc : 0
    () : groups() empty tuple
    {} : groupdict() empty dict
    """