import os
import sys
import copy
import random
import numpy as np
import time
from tqdm import tqdm
import itertools
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import torchvision.transforms as transforms
from torchvision.utils import save_image

import coverage
import utility
from style_operator import Stylized
import image_transforms
from torchvision.models import resnet50, ResNet50_Weights

from torch.utils.data import Dataset
import torch

import hashlib

def hash_numpy(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return h.hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Parameters(object):
    def __init__(self, base_args):
        self.model = base_args.model
        self.dataset = base_args.dataset
        self.criterion = base_args.criterion
        self.use_sc = self.criterion in ['LSC', 'DSC', 'MDSC']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4

        self.batch_size = 32
        # CHANGED: a "batch" is now the set of N seed inputs that share one perturbation.
        self.num_per_perturbation = getattr(base_args, 'num_per_perturbation', 8)
        self.mutate_batch_size = self.num_per_perturbation
        self.nc = 3
        self.image_size = 224 if self.dataset == 'ImageNet' else 32
        self.input_shape = (1, self.image_size, self.image_size, 3)
        self.num_class = 1000 if self.dataset == 'ImageNet' else 10
        self.num_per_class = None # All images in the set

        # NEW: maximum number of operators in a perturbation sequence.
        self.max_seq_len = getattr(base_args, 'max_seq_len', 5)

        self.input_scale = 255
        self.noise_data = False
        self.K = 64
        self.batch1 = 64
        self.batch2 = 16

        self.alpha = 0.2 # default 0.02
        self.beta = 0.5 # default 0.2
        self.TRY_NUM = 50
        self.save_every = 1
        self.output_dir = './data/output/Coverage/Fuzzer/'

        translation = list(itertools.product([getattr(image_transforms, "image_translation")],
                                            [(-5, -5), (-5, 0), (0, -5), (0, 0), (5, 0), (0, 5), (5, 5)]))
        scale = list(itertools.product([getattr(image_transforms, "image_scale")], list(np.arange(0.8, 1, 0.05))))
        # shear = list(itertools.product([getattr(image_transforms, "image_shear")], list(range(-3, 3))))
        rotation = list(itertools.product([getattr(image_transforms, "image_rotation")], list(range(-30, 30))))

        contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [0.8 + 0.2 * k for k in range(7)]))
        brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], [10 + 10 * k for k in range(7)]))
        blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k + 1 for k in range(10)]))

        self.stylized = Stylized(self.image_size)

        self.G = translation + scale + rotation #+ shear
        self.P = contrast + brightness + blur
        self.S = list(itertools.product([self.stylized.transform], [0.4, 0.6, 0.8]))
        self.save_batch = False

class INFO(object):
    def __init__(self):
        self.dict = {}

    def __getitem__(self, i):
        _i = str(i)
        if _i in self.dict:
            return self.dict[_i]
        else:
            I0, state = i, 0
            return I0, state

    def __setitem__(self, i, s):
        _i = str(i)
        self.dict[_i] = s
        return self.dict[_i]

class Fuzzer:
    def __init__(self, params, criterion):
        self.params = params
        self.epoch = 0
        self.time_slot = 60 * 10
        self.time_idx = 0
        self.info = INFO()
        self.hyper_params = {
            'alpha': 0.4, # [0, 1], default 0.02, 0.1 # number of pix
            'beta': 0.8, # [0, 1], default 0.2, 0.5 # max abs pix
            'TRY_NUM': 50,
            'p_min': 0.01,
            'gamma': 5,
            'K': 64
        }
        self.logger = utility.Logger(params, self)
        self.criterion = criterion
        self.initial_coverage = copy.deepcopy(criterion.current)
        self.delta_time = 0
        self.delta_batch = 0
        self.num_ae = 0
        self.orig_map = {}

    def exit(self):
        self.print_info()
        self.criterion.save(self.params.coverage_dir + 'coverage.pt')
        self.logger.exit()

    def can_terminate(self):
        return self.delta_time > 5 * 60

    def print_info(self):
        self.logger.update(self)

    def is_adversarial(self, image, label, k=1):
        with torch.no_grad():
            scores = self.criterion.model(image)
            _, ind = scores.topk(k, dim=1, largest=True, sorted=True)
            correct = ind.eq(label.view(-1, 1).expand_as(ind))
            wrong = ~correct
            index = (wrong == True).nonzero(as_tuple=True)[0]
            wrong_total = wrong.view(-1).float().sum()
            return wrong_total, index, ind.squeeze(1)

    def to_batch(self, data_list):
        batch_list = []
        batch = []
        for i, data in enumerate(data_list):
            if i and i % self.params.mutate_batch_size == 0:
                batch_list.append(np.stack(batch, 0))
                batch = []
            batch.append(data)
        if len(batch):
            batch_list.append(np.stack(batch, 0))
        return batch_list

    def image_to_input(self, image):
        scaled_image = image / self.params.input_scale
        tensor_image = torch.from_numpy(scaled_image).transpose(1, 3)
        normalized_image = utility.image_normalize(tensor_image, self.params.dataset)
        return normalized_image

    # ------------------------------------------------------------------ #
    # CHANGED: perturbation-centric main loop.
    # We keep the seed batches (Bs) FIXED and, for each batch, evolve one
    # shared perturbation (a sequence of ops) applied to all N inputs.
    # ------------------------------------------------------------------ #
    def run(self, I_input, L_input):
        T = self.Preprocess(I_input, L_input)
        B_c, Bs, Bs_label = T

        # NEW: one perturbation (op sequence, initially empty) per seed batch.
        self.perturbations = [[] for _ in Bs]

        del I_input
        del L_input
        gc.collect()

        B, B_label, B_id = self.SelectNext(T)
        self.epoch = 0
        start_time = time.time()
        overall_counts = [0]
        delta_times = [0]
        coverage_gains = [self.criterion.current.item() if isinstance(self.criterion.current, torch.Tensor) else self.criterion.current]

        while not self.can_terminate():
            if self.epoch % 100 == 0:
                self.print_info()

            pert = self.perturbations[B_id]

            # Try to find a perturbation mutation that increases coverage when
            # applied to the WHOLE seed batch (cross-input feedback).
            accepted = False
            torch_image = None
            torch_label = None
            for _ in range(self.hyper_params['TRY_NUM']):
                new_pert = self.MutatePerturbation(pert)
                mutated = self.apply_perturbation(B, new_pert)
                if mutated is None:                 # perceptually invalid for some input
                    continue
                if not self.isChanged(B, mutated):  # no actual change
                    continue

                torch_image = self.image_to_input(mutated).to(self.params.device)
                torch_label = torch.from_numpy(np.array(B_label)).to(self.params.device)

                if self.params.use_sc:
                    cov_dict = self.criterion.calculate(torch_image, torch_label)
                else:
                    cov_dict = self.criterion.calculate(torch_image)
                gain = self.criterion.gain(cov_dict)

                if self.CoverageGain(gain):
                    self.criterion.update(cov_dict, gain)
                    self.perturbations[B_id] = new_pert   # keep the improved perturbation
                    accepted = True
                    break

            if accepted:
                self.delta_batch += 1
                self.BatchPrioritize(T, B_id)

                num_wrong, ae_index, mutated_labels = self.is_adversarial(torch_image, torch_label)
                if num_wrong > 0:
                    self.num_ae += num_wrong
                    ae_indices = ae_index.tolist()
                    mutated_inv = utility.image_normalize_inv(torch_image, self.params.dataset)
                    for i, idx in enumerate(ae_indices):
                        ground_truth = B_label[idx]
                        mutated_label = mutated_labels[idx].item()
                        id = f"{self.epoch}_{i}"
                        os.makedirs(f"{self.params.image_dir}/aes/{ground_truth}/", exist_ok=True)
                        os.makedirs(f"{self.params.image_dir}/orig/{ground_truth}/", exist_ok=True)

                        save_image(mutated_inv[idx].data, f"{self.params.image_dir}/aes/{ground_truth}/{id}_ae_{mutated_label}.jpg")

                        # The original seed is fixed, so we can save it directly.
                        old_image = np.expand_dims(B[idx], axis=0)
                        old_image = self.image_to_input(old_image)
                        old_image = utility.image_normalize_inv(old_image, self.params.dataset)
                        save_image(old_image[0].data, f"{self.params.image_dir}/orig/{ground_truth}/{id}_orig_{ground_truth}.png", normalize=True, format='PNG')

            gc.collect()

            B, B_label, B_id = self.SelectNext(T)
            self.epoch += 1
            self.delta_time = time.time() - start_time
            delta_times.append(self.delta_time)
            overall_counts.append(self.num_ae.item() if isinstance(self.num_ae, torch.Tensor) else self.num_ae)
            coverage_gains.append(self.criterion.current.item() if isinstance(self.criterion.current, torch.Tensor) else self.criterion.current)

        with open(f"{self.params.image_dir}/statistics.json", "w") as f:
            json.dump({
                "time": delta_times,
                "coverage": coverage_gains,
                "overall_ae_counts": overall_counts
            }, f, indent=4)

    def Preprocess(self, image_list, label_list):
        randomize_idx = np.arange(len(image_list))
        np.random.shuffle(randomize_idx)
        image_list = [image_list[idx] * self.params.input_scale for idx in randomize_idx]
        label_list = [label_list[idx] for idx in randomize_idx]

        Bs = self.to_batch(image_list)
        Bs_label = self.to_batch(label_list)

        return list(np.zeros(len(Bs))), Bs, Bs_label

    def calc_priority(self, B_ci):
        if B_ci < (1 - self.hyper_params['p_min']) * self.hyper_params['gamma']:
            return 1 - B_ci / self.hyper_params['gamma']
        else:
            return self.hyper_params['p_min']

    def SelectNext(self, T):
        B_c, Bs, Bs_label = T
        B_p = [self.calc_priority(B_c[i]) for i in range(len(B_c))]
        c = np.random.choice(len(Bs), p=B_p / np.sum(B_p))
        return Bs[c], Bs_label[c], c

    def isChanged(self, I, I_new):
        return np.any(I != I_new)

    def CoverageGain(self, gain):
        if gain is not None:
            if isinstance(gain, tuple):
                return gain[0] > 0
            else:
                return gain > 0
        else:
            return False

    def BatchPrioritize(self, T, B_id):
        B_c, Bs, Bs_label = T
        B_c[B_id] += 1

    # ------------------------------------------------------------------ #
    # CHANGED: Mutate now operates on a PERTURBATION (op sequence), not an image.
    # It extends (or, at max length, replaces) one operator. The "one geometric
    # op, then only P+S" rule from the original INFO state machine is preserved.
    # ------------------------------------------------------------------ #
    def MutatePerturbation(self, perturbation):
        G, P, S = self.params.G, self.params.P, self.params.S
        has_geo = any(op in G for op in perturbation)
        candidate_ops = (P + S) if has_geo else (G + P + S)
        t, p = self.randomPick(candidate_ops)

        new_pert = list(perturbation)
        if len(new_pert) >= self.params.max_seq_len:
            new_pert[np.random.randint(len(new_pert))] = (t, p)   # replace
        else:
            new_pert.append((t, p))                                # extend
        return new_pert

    # ------------------------------------------------------------------ #
    # NEW: apply the same op sequence to every image in the seed batch, with
    # the same perceptual-validity check f() used by the original (computed
    # against the geometric-only baseline, mirroring the old I0_G logic).
    # Returns None if the perturbation is invalid for any input.
    # ------------------------------------------------------------------ #
    def apply_perturbation(self, images, perturbation):
        geo_ops = [(t, p) for (t, p) in perturbation if (t, p) in self.params.G]
        out = []
        for I in images:
            J = I
            for (t, p) in perturbation:
                J = np.clip(t(J, p).reshape(*(self.params.input_shape[1:])), 0, 255)
            base = I
            for (t, p) in geo_ops:
                base = np.clip(t(base, p).reshape(*(self.params.input_shape[1:])), 0, 255)
            if not self.f(base, J):
                return None
            out.append(J.astype('float32'))
        return np.stack(out, 0).astype('float32')

    def randomPick(self, A):
        c = np.random.randint(0, len(A))
        return A[c]

    def f(self, I, I_new):
        if (np.sum((I - I_new) != 0) < self.hyper_params['alpha'] * np.sum(I > 0)):
            return np.max(np.abs(I - I_new)) <= 255
        else:
            return np.max(np.abs(I - I_new)) <= self.hyper_params['beta'] * 255


if __name__ == '__main__':
    import os
    import argparse
    import torchvision
    import gc

    import utility
    import models
    import tool
    import coverage
    import constants
    import data_loader

    import signal
    def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            try:
                if engine is not None:
                    engine.print_info()
                    if engine.logger is not None:
                        engine.logger.exit()
                    if engine.criterion is not None:
                        engine.criterion.save(args.coverage_dir + 'coverage_int.pth')
            except:
                pass
            sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                            choices=['CIFAR10', 'ImageNet'])
    parser.add_argument('--model', type=str, default='resnet50',
                            choices=['resnet50', 'vgg16_bn', 'mobilenet_v2'])
    parser.add_argument('--criterion', type=str, default='NLC',
                            choices=['NLC', 'NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'CC',
                    'LSC', 'DSC', 'MDSC'])
    parser.add_argument('--output_dir', type=str, default='./test_folder')
    parser.add_argument('--random_seed', type=int, default=0)
    # NEW: perturbation-centric options.
    parser.add_argument('--num_per_perturbation', type=int, default=8,
                            help='N: number of seed inputs sharing one perturbation.')
    parser.add_argument('--max_seq_len', type=int, default=5,
                            help='Maximum number of operators in a perturbation sequence.')
    base_args = parser.parse_args()

    args = Parameters(base_args)
    set_seed(base_args.random_seed)
    args.exp_name = ('%s-%s-%s' % (args.dataset, args.model, args.criterion))
    print(args.exp_name)
    utility.make_path(args.output_dir)
    utility.make_path(args.output_dir + args.exp_name)

    args.image_dir = args.output_dir + args.exp_name + '/image/'
    args.coverage_dir = args.output_dir + args.exp_name + '/coverage/'
    args.log_dir = args.output_dir + args.exp_name + '/log/'

    utility.make_path(args.image_dir)
    utility.make_path(args.coverage_dir)
    utility.make_path(args.log_dir)

    if args.dataset == 'ImageNet':
        if args.model == "resnet50":
            imagenet_mean = (0.485, 0.456, 0.406)
            imagenet_std  = (0.229, 0.224, 0.225)
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = torchvision.models.__dict__[args.model](pretrained=False)
            path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (args.dataset, args.model)))
            model.load_state_dict(torch.load(path))

        assert args.image_size == 224
        assert args.num_class <= 1000
    elif args.dataset == 'CIFAR10':
        model = getattr(models, args.model)(pretrained=False)
        path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pt' % (args.dataset, args.model)))
        model.load_state_dict(torch.load(path))
        assert args.image_size == 32
        assert args.num_class <= 10


    model.to(args.device)
    model.eval()

    input_size = (1, args.nc, args.image_size, args.image_size)
    random_data = torch.randn(input_size).to(args.device)
    layer_size_dict = tool.get_layer_output_sizes(model, random_data)


    if args.dataset == 'CIFAR10':
        data_set = data_loader.CIFAR10FuzzDataset(args, split='test')
    elif args.dataset == 'ImageNet':
        data_set = data_loader.ImageNetFuzzDataset(args, split='val')
    TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = data_loader.get_loader(args)

    image_list, label_list = data_set.build(only_correct=True, model=model)
    image_numpy_list = data_set.to_numpy(image_list)
    label_numpy_list = data_set.to_numpy(label_list, False)
    print("Filtered Dataset Length: ", len(image_numpy_list))

    del image_list
    del label_list
    gc.collect()

    hyper_map = {
        'NLC': None,
        'NC': 0,
        'KMNC': 100,
        'SNAC': None,
        'NBC': None,
        'TKNC': 10,
        'TKNP': 50,
        'CC': 10 if args.dataset == 'CIFAR10' else 1000,
        'LSA': 10,
        'DSA': 0.1,
        'MDSA': 10
    }

    if args.use_sc:
        criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=hyper_map[args.criterion], min_var=1e-5, num_class=TOTAL_CLASS_NUM)
    else:
        criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=hyper_map[args.criterion])

    criterion.build(seed_loader)
    if args.criterion not in ['CC', 'TKNP', 'LSC', 'DSC', 'MDSC']:
        criterion.assess(seed_loader)
    '''
    For LSC/DSC/MDSC/CC/TKNP, initialization with training data is too slow (sometimes may
    exceed the memory limit). You can skip this step to speed up the experiment, which
    will not affect the conclusion because we only compare the relative order of coverage
    values, rather than the exact numbers.
    '''

    initial_coverage = copy.deepcopy(criterion.current)
    print('Initial Coverage: %f' % initial_coverage)
    engine = Fuzzer(args, criterion)
    engine.run(image_numpy_list, label_numpy_list)
    engine.exit()
    