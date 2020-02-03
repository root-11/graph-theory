import cProfile
import io
import pstats


def profileit(func):
    """ decorator for profiling function. """
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".txt"  # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open(datafn, 'w') as perf_file:
            perf_file.write(s.getvalue())
        return retval

    return wrapper
