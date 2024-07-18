from numpy.random.mtrand import sample
from util import *


class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''

    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28, 28], is_top=False, n_labels=10,
                 batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end.
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """

        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom: self.image_size = image_size

        self.is_top = is_top

        if is_top: self.n_labels = 10

        self.batch_size = batch_size

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = 0.01

        self.momentum = 0.7

        # self.print_period = 5000
        self.print_period = 5000

        self.store_recon_loss = []

        self.rf = {  # receptive-fields. Only applicable when visible layer is input data
            "period": 5000,  # iteration period to visualize
            "grid": [5, 5],  # size of the grid
            "ids": np.random.randint(0, self.ndim_hidden, 25)  # pick some random hidden units
        }

        return

    def cd1(self, visible_trainset, n_iterations=10000):

        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("learning CD1")

        n_samples = visible_trainset.shape[0]

        rng = np.random.default_rng()
        # Group data into sizes of mini-batches
        minibatch_idx = rng.permutation(n_samples).reshape(-1, self.batch_size)
        # in case n_iterations > n_samples, we cycle through the permutations again
        for i in range(int(np.ceil(n_iterations / (n_samples / self.batch_size))) - 1):
            minibatch_idx = np.concatenate((minibatch_idx, rng.permutation(n_samples).reshape(-1, self.batch_size)))

        for it in range(n_iterations):
            # Generate visible_minibatch from the visible_trainset
            visible_minibatch = visible_trainset[minibatch_idx[it], :]

            # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
            # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
            # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.

            h0_probabilities, h0_activity = self.get_h_given_v(visible_minibatch)
            v1_probabilities, v1_activity = self.get_v_given_h(h0_activity)
            h1_probabilities, h1_activity = self.get_h_given_v(v1_activity)

            # update the parameters using function 'update_params'
            # YM: I used probabilities, though in principle, using activities should also work
            self.update_params(v_0=visible_minibatch, h_0=h0_probabilities, v_k=v1_probabilities, h_k=h1_probabilities)

            # visualize once in a while when visible layer is input images

            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape((self.image_size[0], self.image_size[1], -1)),
                       it=it, grid=self.rf["grid"])

            # print progress

            if it % self.print_period == 0 or it % 9999 == 0:
                # h0_probabilities, h0_activity = self.get_h_given_v(visible_trainset)
                # v1_probabilities, v1_activity = self.get_v_given_h(h0_activity)

                rec_loss = np.linalg.norm(visible_minibatch - v1_activity)
                self.store_recon_loss.append(rec_loss)
                print("iteration=%7d recon_loss=%4.4f" % (it, rec_loss))

            # print("iteration=%7d"%(it))

        # store_recon_loss = np.array(self.store_recon_loss)
        # np.savetxt('500_store_recon_loss.csv', store_recon_loss)
        return

    def update_params(self, v_0, h_0, v_k, h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters

        self.delta_bias_v = np.mean(v_0 - v_k, axis=0)

        self.delta_bias_h = np.mean(h_0 - h_k, axis=0)

        # self.delta_weight_vh = np.zeros((self.ndim_visible, self.ndim_hidden))

        self.delta_weight_vh = np.matmul(np.matrix.transpose(v_0), h_0) / self.batch_size - np.matmul(
            np.matrix.transpose(v_k), h_k) / self.batch_size

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h

        return

    def get_h_given_v(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses undirected weight "weight_vh" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]

        # compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below)

        summed_func = np.matmul(visible_minibatch, self.weight_vh)
        summed_func += self.bias_h

        probability = sigmoid(summed_func)
        hidden_activations = sample_binary(probability)

        return probability, hidden_activations

    def get_v_given_h(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]
        probability = np.zeros((n_samples, self.ndim_visible))
        visible_activations = np.zeros((n_samples, self.ndim_visible))

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.

            summed_func = np.matmul(hidden_minibatch, self.weight_vh.T)
            summed_func += self.bias_v

            prob1 = summed_func[:, :-10]
            prob2 = summed_func[:, -10:]

            prob1 = sigmoid(prob1)
            prob2 = softmax(prob2)
            visible_activations1 = sample_binary(prob1)
            visible_activations2 = sample_categorical(prob2)

            probability = np.concatenate((prob1, prob2), axis=1)
            visible_activations = np.concatenate((visible_activations1, visible_activations2), axis=1)

            return probability, visible_activations


        else:

            # compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)

            summed_func = np.matmul(hidden_minibatch, self.weight_vh.T)
            summed_func += self.bias_v

            probability = sigmoid(summed_func)
            visible_activations = sample_binary(probability)

        return probability, visible_activations

    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):

        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(np.transpose(self.weight_vh))
        self.weight_vh = self.weight_vh

    def get_h_given_v_dir(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below)

        summed_func = np.matmul(visible_minibatch, self.weight_v_to_h)
        summed_func += self.bias_h

        probability = sigmoid(summed_func)
        hidden_activations = sample_binary(probability)

        return probability, hidden_activations

    def get_v_given_h_dir(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]
        probability = np.zeros((n_samples, self.ndim_visible))
        visible_activations = np.zeros((n_samples, self.ndim_visible))

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.

            # MEMO (Tai): same with else
            summed_func = np.matmul(hidden_minibatch, self.weight_vh.T)
            summed_func += self.bias_v

            prob1 = summed_func[:, :-10]
            prob2 = summed_func[:, -10:]

            prob1 = sigmoid(prob1)
            prob2 = softmax(prob2)
            visible_activations1 = sample_binary(prob1)
            visible_activations2 = sample_categorical(prob2)

            probability = np.concatenate((prob1, prob2), axis=1)
            visible_activations = np.concatenate((visible_activations1, visible_activations2), axis=1)

            return probability, visible_activations

        else:

            # compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)

            summed_func = np.matmul(hidden_minibatch, self.weight_h_to_v)
            summed_func += self.bias_v

            probability = sigmoid(summed_func)
            visible_activations = sample_binary(probability)

        return probability, visible_activations

    def update_generate_params(self, inps, trgs, preds):

        """Update generative weight "weight_h_to_v" and bias "bias_v"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v

        return

    def update_recognize_params(self, inps, trgs, preds):

        """Update recognition weight "weight_v_to_h" and bias "bias_h"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h

        return    
