import dill

def run_dill_encoded(what):
    """
    Use dill as replacement for pickle to enable multiprocessing on instancemethods
    """
    fun, args = dill.loads(what)
    return fun(*args)

def apply_async(pool, fun, args, callback):
    """
    Wrapper around apply_async() from multiprocessing, to use dill instead of pickle.
    This is a workaround to enable multiprocessing of classes.
    """
    return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),), callback = callback)

