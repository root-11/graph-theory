import cProfile
import io
import pstats


def profileit(func):
    """ decorator for profiling function.
    usage:
    >>> def this(var1, var2):
            # do something

    >>> new_this = count_calls(this)
    >>> calls, cprofile_text = new_this(var1=1,var2=2)
    """
    def wrapper(*args, **kwargs):
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
        ps.print_stats()
        text = s.getvalue()
        calls = int(text.split('\n')[0].lstrip().split(" ")[0])
        return calls, text
    return wrapper
