def get_model(model_name, args):
    name = model_name.lower()
    if name=="adapter":
        from models.RAA import Learner
        return Learner(args)
    else:
        assert 0
