import theano.tensor as T


class CrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, inputs, labels):
        # Your codes here
        # hint: labels are already in one-hot form
        return -T.sum(labels * T.log (inputs))
