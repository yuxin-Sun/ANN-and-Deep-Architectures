from util import *
from rbm import RestrictedBoltzmannMachine


class LightDeepBeliefNet():
    '''
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <--->[hid] ---> [vis]
    top : top
    hid : hidden
    vis : visible
    '''

    def __init__(self, sizes, image_size, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {

            'vis--hid': RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                   is_bottom=True, image_size=image_size, batch_size=batch_size),

            'hid--top': RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["top"],
                                                   batch_size=batch_size, is_top=True),

        }

        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size

        self.print_period = 2000

        return

    def train_greedylayerwise(self, vis_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack.
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try:

            self.loadfromfile_rbm(loc="light_trained_rbm", name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()

            self.loadfromfile_rbm(loc="light_trained_rbm", name="hid--top")
            self.rbm_stack["hid--top"].untwine_weights()

        except IOError:

            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily

            print("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """
            ## Tai : use cd1
            self.rbm_stack["vis--hid"].cd1(visible_trainset=vis_trainset, n_iterations=n_iterations)
            self.rbm_stack["vis--hid"].untwine_weights()
            self.savetofile_rbm(loc="trained_rbm", name="vis--hid")

            print("training hid--top")
            """ 
            CD-1 training for hid--top 
            """
            ## Tai : apply cd1 for hid-pen layers
            next_train_set = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)
            self.rbm_stack["hid--top"].cd1(visible_trainset=next_train_set[1], n_iterations=n_iterations)
            self.rbm_stack["hid--top"].untwine_weights()
            self.savetofile_rbm(loc="trained_rbm", name="hid--top")

        return

    def get_reconstruction_error(self, vis_trainset):
        output = self.step(vis_trainset)
        rec_loss = ((vis_trainset - output) ** 2).mean()

        print(rec_loss)

    def step(self, batch):
        # Forward
        _, hid_act = self.rbm_stack["vis--hid"].get_h_given_v_dir(batch)
        _, top_act = self.rbm_stack["hid--top"].get_h_given_v_dir(hid_act)

        # Backward

        _, hid_act = self.rbm_stack["hid--top"].get_v_given_h_dir(top_act)
        _, vis_act = self.rbm_stack["vis--hid"].get_v_given_h_dir(hid_act)

        return vis_act

    def loadfromfile_rbm(self, loc, name):

        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_rbm(self, loc, name):

        np.save("%s/rbm.%s.weight_vh" % (loc, name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return

    def loadfromfile_dbn(self, loc, name):

        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy" % (loc, name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/dbn.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/dbn.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_dbn(self, loc, name):

        np.save("%s/dbn.%s.weight_v_to_h" % (loc, name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v" % (loc, name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return

