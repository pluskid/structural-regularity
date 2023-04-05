import os
import itertools

import numpy as np
import numpy.random as npr

from tqdm import tqdm
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax.config import config
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Flatten, Dense, Relu, LogSoftmax

import tensorflow_datasets as tfds


def one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))


def batch_correctness(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return predicted_class == target_class


def load_mnist():
  # NOTE: even when we specify shuffle_files=False, the examples 
  # in the MNIST dataset loaded by tfds is still NOT in the 
  # original MNIST data order
  raw_data = tfds.load(name='mnist', batch_size=-1,
                       as_dataset_kwargs={'shuffle_files': False})
  raw_data = tfds.as_numpy(raw_data)
  train_byte_images = raw_data['train']['image']
  train_images = train_byte_images.astype(np.float32) / 255
  train_int_labels = raw_data['train']['label']
  train_labels = one_hot(train_int_labels, 10)
  return dict(train_images=train_images, train_labels=train_labels,
              train_byte_images=train_byte_images, 
              train_int_labels=train_int_labels)


init_random_params, predict = stax.serial(
    Flatten,
    Dense(512), Relu,
    Dense(256), Relu,
    Dense(10), LogSoftmax)
mnist_data = load_mnist()


def subset_train(seed, subset_ratio):
  jrng = random.PRNGKey(seed)
  
  step_size = 0.1
  num_epochs = 10
  batch_size = 128
  momentum_mass = 0.9

  num_train_total = mnist_data['train_images'].shape[0]
  num_train = int(num_train_total * subset_ratio)
  num_batches = int(np.ceil(num_train / batch_size))

  rng = npr.RandomState(seed)
  subset_idx = rng.choice(num_train_total, size=num_train, replace=False)
  train_images = mnist_data['train_images'][subset_idx]
  train_labels = mnist_data['train_labels'][subset_idx]

  def data_stream(shuffle=True):
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  batches = data_stream()

  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  _, init_params = init_random_params(jrng, (-1, 28 * 28))
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  for epoch in range(num_epochs):
    for _ in range(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))

  params = get_params(opt_state)
  trainset_correctness = batch_correctness(
    params, (mnist_data['train_images'], mnist_data['train_labels']))
  trainset_mask = np.zeros(num_train_total, dtype=bool)
  trainset_mask[subset_idx] = True
  return trainset_mask, np.asarray(trainset_correctness)


def estimate_cscores():
  n_runs = 200
  subset_ratios = np.linspace(0.1, 0.9, 9)
  
  cscores = {}
  for ss_ratio in subset_ratios:
    results = []
    for i_run in tqdm(range(n_runs), desc=f'SS Ratio={ss_ratio:.2f}'):
      results.append(subset_train(i_run, ss_ratio))

    trainset_mask = np.vstack([ret[0] for ret in results])
    trainset_correctness = np.vstack([ret[1] for ret in results])
    inv_mask = np.logical_not(trainset_mask)

    cscores[ss_ratio] = (np.sum(trainset_correctness * inv_mask, axis=0) / 
                         np.maximum(np.sum(inv_mask, axis=0), 1e-10))
  
  return cscores
  

def show_examples(cscores, n_show=10):
  def show_image(ax, image, vmin=None, vmax=None, title=None):
    if image.ndim == 3 and image.shape[2] == 1:
      image = image.reshape((image.shape[0], image.shape[1]))
    ax.axis('off')
    ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    if title is not None:
      ax.set_title(title, fontsize='x-small')

  n_classes = 10
  fig, axs = plt.subplots(nrows=n_classes, ncols=2*n_show+1, 
                          figsize=(2*n_show+1, n_classes))

  for klass in range(n_classes):
    idx_klass = np.nonzero(mnist_data['train_int_labels'] == klass)[0]
    idx_sorted = np.argsort(-cscores[idx_klass])
    for i in range(n_show):
      idx = idx_klass[idx_sorted[i]]
      show_image(axs[klass, i], mnist_data['train_byte_images'][idx])

      idx = idx_klass[idx_sorted[-(i+1)]]
      show_image(axs[klass, i+n_show+1], mnist_data['train_byte_images'][idx])

    for i in range(2*n_show+1):
      axs[klass, i].axis('off')

  plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0, right=1, bottom=0, top=1)
  plt.savefig('mnist-examples.pdf', bbox_inches='tight')
  
  
if __name__ == '__main__':
  npz_fn = 'mnist-cscores.npy'
  if os.path.exists(npz_fn):
    cscores = np.load(npz_fn)
  else:
    cscores = estimate_cscores()
    cscores = np.mean([x for x in cscores.values()], axis=0)
    np.save(npz_fn, cscores)

  show_examples(cscores)
