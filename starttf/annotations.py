class RunOnce(object):
    def __init__(self, f):
        self.f = f
        self.called = False
    
    def __call__(self, *args, **kwargs):
        if not self.called:
            self.called = True
            self.f(*args, **kwargs)
