OPTIONS = {
    'use_dask': False
}


class set_options:
    """Set options for trendfit in a controlled context.

    Currently supported options:

    - ``use_dask``: Enable dask for parallel operations where supported.
      Default: ``False``.

    You can use ``set_options`` either as a context manager or to set
    global options.

    """

    def __init__(self, **kwargs):
        self.old = {}

        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    "argument name {%r} is not in the set of valid options {%r}"
                    .format(k, set(OPTIONS))
                )
            self.old[k] = OPTIONS[k]

        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)
